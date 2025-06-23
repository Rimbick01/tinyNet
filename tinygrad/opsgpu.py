import numpy as np
from .tensor import Function, register
import pyopencl as cl

# ************* basic ops *************

#def buffer(ctx, x):
#  sz=4
#  for s in shape:
#    sz*=s
#  res_g = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY, sz)
#  res_g.shape = shape
#  return res_g
#
#def buffer_like(ctx, x):
#  return buffer(ctx,x.shape)
#
class Add(Function):
  @staticmethod
  def forward(ctx, x, y):
    res_g = cl.Buffer(ctx.cl_ctx, cl.mem_flags.WRITE_ONLY, x.size)
    res_g.shape = x.shape
    #ret = buffer_like(ctx, x)
    prg = cl.Program(ctx.cl_ctx, """
    __kernel void sum(
        __global const float *a_g, __global const float *b_g, __global float *res_g)
    {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] + b_g[gid];
    }
    """).build()
    prg.sum(ctx.cl_queue, [x.size//4], None, x, y, res_g)
    #prg.sum(ctx.cl_queue, [x.size//4], None, x, y, ret)
    #return ret
    return res_g

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, grad_output
register('add', Add, gpu=True)
     
#class Dot(function):
#  @staticmethod
#  def forward(ctx, x, y):
#    pass
