

def update_cast_joint_errors(q, limits, eps=1e-6):
    below_lower = q < limits[:, 0]
    above_upper = q > limits[:, 1]

    q[below_lower] += eps
    q[above_upper] -= eps
    return q


from mogen.Generation.parameter import init_par


# def check_consistency(robot,
#                       db_file=None,
#                       q=None, f=None, o=None, img=None):
#
#     gen = init_par(robot_id=robot.id)
#
#     if q is None:
#         i_w, i_s, q0, q, o, f = sql2.get_values_sql(file=db_file, table='paths',
#                                                     rows=i,
#                                                     columns=['world_i32', 'sample_i32', 'q0_f32', 'q_f32',
#                                                              'objective_f32', 'feasible_f32'],
#                                                     values_only=True)
#
#     if img is None:
#         img_cmp = sql2.get_values_sql(file=db_file, rows=np.arange(300), table='worlds', columns='img_cmp', values_only=True)
#         img = compressed2img(img_cmp=img_cmp, shape=gen.par.world.shape, dtype=bool)
#
#     q = q.reshape(-1, gen.par.n_wp, robot.n_dof).copy()
#
#     o_label, f_label = objective_feasibility(q=q, imgs=img, par=gen.par, iw=i_w)
#     print(f.sum(), f_label.sum(), (f == f_label).mean())
