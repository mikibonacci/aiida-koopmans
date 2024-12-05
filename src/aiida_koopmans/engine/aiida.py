from koopmans.engines.engine import Engine
from koopmans.step import Step
from koopmans.calculators import Calc
from koopmans.pseudopotentials import read_pseudo_file
from koopmans.status import Status

from aiida.engine import run_get_node, submit

from aiida_koopmans.utils import *

from aiida_pseudo.data.pseudo import UpfData

import time

import dill as pickle

from aiida import orm, load_profile
load_profile()

class AiiDAEngine(Engine):

    """
    Step data is a dictionary containing the following information:
    step_data = {calc.directory: {'workchain': workchain, 'remote_folder': remote_folder}}
    and any other info we need for AiiDA.
    """
    def __init__(self, *args, **kwargs):
        self.blocking = kwargs.pop('blocking', True)
        self.step_data = { # TODO: change to a better name
            'configuration': kwargs.pop('configuration', None),
            'steps': {}
            }
        # here we add the logic to populate configuration by default
        # 1. we look for codes stored in AiiDA at localhost, e.g. pw-version@localhost,
        # 2. we look for codes in the PATH,
        # 3. if we don't find the code in AiiDA db but in the PATH, we store it in AiiDA db.
        # 4. if we don't find the code in AiiDA db and in the PATH and not configuration is provided, we raise an error.
        if self.step_data['configuration'] is None:
            raise NotImplementedError("Configuration not provided")

        # 5. if no resource info in configuration, we try to look at PARA_PREFIX env var.


        super().__init__(*args, **kwargs)

    def run(self, step: Step):

        self.get_status(step)
        if step.prefix in ['wannier90_preproc', 'pw2wannier90']:
            self.set_status(step, Status.COMPLETED)
            return

        self.step_data['steps'][step.uid] = {} # maybe not needed
        builder, self.step_data = get_builder_from_ase(calculator=step, step_data=self.step_data) # ASE to AiiDA conversion. put some error message if the conversion fails
        running = submit(builder)
        print(f"Running workchain {running.pk} for step {step.uid}")
        # running = aiidawrapperwchain.submit(builder) # in the non-blocking case.
        
        # The below will be passed to the context, so we will need to store also the instance of the submitted workchain, if in KoopmansWorkChain.
        self.step_data['steps'][step.uid] = {'workchain': running.pk, } #'remote_folder': running.outputs.remote_folder}

        self.set_status(step, Status.RUNNING)

        return

    def load_step_data(self):
        try:
            with open('step_data.pkl', 'rb') as f:
                # this will overwrite the step_data[configuration],
                # i.e. if we change codes or res we will not see it if
                # the file already exists.
                self.step_data = pickle.load(f)
        except FileNotFoundError:
            pass

    def dump_step_data(self):
        with open('step_data.pkl', 'wb') as f:
            pickle.dump(self.step_data, f)

    def get_status(self, step: Step) -> Status:
        status = self.get_status_by_uid(step.uid)
        #print(f"Getting status for step {step.uid}: {status}")
        return status
        
    
    def get_status_by_uid(self, uid: str) -> Status:
        self.load_step_data()
        if uid not in self.step_data['steps']:
            self.step_data['steps'][uid] = {'status': Status.NOT_STARTED}
        return self.step_data['steps'][uid]['status']

    def set_status(self, step: Step, status: Status):
        self.set_status_by_uid(step.uid, status)
        
    def set_status_by_uid(self, uid: str, status: Status):
        self.step_data['steps'][uid]['status'] = status
        self.dump_step_data()

    def update_statuses(self) -> None:
        
        time.sleep(1)
        for uid in self.step_data['steps']:

            if not self.get_status_by_uid(uid) == Status.RUNNING:
                continue

            workchain = orm.load_node(self.step_data['steps'][uid]['workchain'])
            if workchain.is_finished_ok:
                self._step_completed_message_by_uid(uid)
                self.set_status_by_uid(uid, Status.COMPLETED)

            elif workchain.is_finished or workchain.is_excepted or workchain.is_killed:
                self._step_failed_message_by_uid(uid)
                self.set_status_by_uid(uid, Status.FAILED)

            return

    def load_results(self, step: Step) -> None:

        self.load_step_data()
        
        if step.prefix in ['wannier90_preproc', 'pw2wannier90']:
            self.set_status(step, Status.COMPLETED)
            return
        workchain = orm.load_node(self.step_data['steps'][step.uid]['workchain'])
        if "remote_folder" in workchain.outputs:
            self.step_data['steps'][step.uid]['remote_folder'] = workchain.outputs.remote_folder.pk
        output = None
        if step.ext_out == ".wout":
            output = read_output_file(step, workchain.outputs.wannier90.retrieved)
        elif step.ext_out in [".pwo",".kho"]:
            output = read_output_file(step, workchain.outputs.retrieved)
            if hasattr(output.calc, 'kpts'):
                step.kpts = output.calc.kpts
        else:
            output = read_output_file(step, workchain.outputs.retrieved)
        if step.ext_out in [".pwo",".wout",".kso",".kho"]:
            step.calc = output.calc
            step.results = output.calc.results
            if step.ext_out == ".pwo": step.generate_band_structure() #nelec=int(workchain.outputs.output_parameters.get_dict()['number_of_electrons']))

            self._step_completed_message(step)

        if step.ext_out in [".pro"]:

            pdos_dir = dump_pdos_outputs(step, workchain.outputs.retrieved)
            prev_dir = step.directory
            step.directory = pdos_dir

            try:
                step.generate_dos()
            except ValueError:
                # ValueError: Must provide energies to create a GridDOSCollection without any DOS data.
                pass
            finally:
                from aiida_koopmans.utils import delete_directory
                delete_directory(pdos_dir.parent)
                step.directory = prev_dir

        self.dump_step_data()

    def load_old_calculator(self, calc: Calc):
        raise NotImplementedError # load_old_calculator(calc)
    
    def get_pseudopotential(self, library: str, element: str):

        qb = orm.QueryBuilder()
        qb.append(orm.Group, filters={'label': {'==': library}}, tag='pseudo_group')
        qb.append(UpfData, filters={'attributes.element': {'==': element}}, with_group='pseudo_group')
        
        pseudo_data = None
        for pseudo in qb.all():
            with tempfile.TemporaryDirectory() as dirpath:
                temp_file = pathlib.Path(dirpath) / (pseudo[0].base.attributes.all['element'] + '.upf')
                with pseudo[0].open(pseudo[0].base.attributes.all['element'] + '.upf', 'rb') as handle:
                    temp_file.write_bytes(handle.read())

                pseudo_data = read_pseudo_file(temp_file)
        
        if not pseudo_data:
            raise ValueError(f"Could not find pseudopotential for element {element} in library {library}")
        
        self.step_data['pseudo_family'] = library
        
        return pseudo_data
