import pymol
from pymol import cmd

pymol.finish_launching()

cmd.load("/home/hebeibei/Work/dl/code/BindSculpt/data/outputs/af3_predictions/output_iteration_0_batch_1749525019/binder_1749525019_0/binder_1749525019_0_model.cif")
cmd.save("/home/hebeibei/Work/dl/code/BindSculpt/data/outputs/af3_predictions/output_iteration_0_batch_1749525019/binder_1749525019_0/binder_1749525019_0_model.pdb")
cmd.quit()
