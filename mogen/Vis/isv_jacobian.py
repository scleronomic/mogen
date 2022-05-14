import numpy as np

from mogen.Vis.InteractiveSampleViewer import InteractiveSampleViewer

from mopla.Optimizer.objectives import o_oc, ik_grad
from mopla.Parameter.parameter import Parameter

from mopla.Optimizer.nullspace import dx2dq_pseudo, nullspace_projection2


class ISV(InteractiveSampleViewer):
    def __init__(self, *, i_world=0, i_sample=0, file=None,
                 par, gd=None):

        super().__init__(i_world=i_world, i_sample=i_sample, file=file,
                         par=par, gd=gd)

        self.drag_end.toggle_visibility(value=True)

    def on_drag(self, *args):
        q = self.drag_start.get_q()

        f0 = self.par.robot.get_frames(q)[self.par.xc.f_idx]
        x, dx_dq = self.par.robot.get_spheres_jac(q)

        self.par.update_oc(img=self.world.img)

        self.par.xc.frame = f0

        # A
        for i in range(100):
            j = ik_grad(par=self.par, q=q, q_close=self.drag_start.get_q()) / 100
            q = nullspace_projection2(robot=self.par.robot, q=q, u=-j, f0=f0, mode='x', f_idx=self.par.xc.f_idx, clip=0.1)

        # B
        # f, j = o_oc.oc_cost_grad(x=x, dx_dq=dx_dq, oc=self.par.oc, jac_only=False)
        # for i in range(100):
        #     j = np.clip(j.sum(axis=0), a_min=-0.1, a_max=+0.1)
        #     q -= j
        #     x, dx_dq = self.par.robot.get_spheres_jac(q)
        #     f, j = o_oc.oc_cost_grad(x=x, dx_dq=dx_dq, oc=self.par.oc, jac_only=False)

        # C
        # b = f > 0
        # if b.sum() > 0:
        #     f = f[b]
        #     j = j[b, :]
        #     dq = dx2dq_pseudo(dx=f, j=-j)
        #     print(f)
        # else:
        #     dq = 0
        # q = q - dq
        print(q)
        self.drag_end.set_q(q)


_par = Parameter(robot='StaticArm04')
_par.n_wp = 1
_par.plan.x_close = False
_par.xc.f_idx = _par.robot.n_frames - 1
_par.weighting.collision = 10
_par.weighting.length = 1

isv = ISV(par=_par)

isv.on_drag()

