import mujoco
print("mujoco", mujoco.__version__, flush=True)
try:
    import dm_control
    print("dm_control", dm_control.__version__, flush=True)
except Exception as e:
    print("dm_control ver?", e, flush=True)
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 64, 64)
r.update_scene(mujoco.MjData(m))
r.render()
import OpenGL.GL as gl
print("GL_RENDERER:", gl.glGetString(gl.GL_RENDERER), flush=True)
print("GL_VENDOR:", gl.glGetString(gl.GL_VENDOR), flush=True)
