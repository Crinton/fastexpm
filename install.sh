dir="/home/hxy/expm/test/fastexpm" #只需要修改路径，就能自动制作python包到指定路径
if [ -d "build" ]; then
  # 如果存在，则删除 build 文件夹
  echo "Directory 'build' exists. Deleting..."
  rm -rf build
  mkdir build && cd build
else
  # 如果不存在，则创建 build 文件夹
  echo "Directory 'build' does not exist. Creating... and Entering build File"
  mkdir build && cd build
fi

cmake .. -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${dir} # ../libs 用户可以自行修改
make -j 4 # 编译并链接
cd "${dir}"
echo "fastexpm exit in $(pwd)/"
echo "start init python package"

touch __init__.py
echo  "from .fastexpm import expm_float,expm_double,expm_complex,expm_doublecomplex
__version__ = '1.0.0'" > __init__.py


touch expm.py
echo "from typing import TypeVar
import numpy as np
import numpy.linalg as _mod_numpy_linalg
LinAlgError = _mod_numpy_linalg.LinAlgError
from .fastexpm import expm_float,expm_double,expm_complex,expm_doublecomplex
Array = TypeVar('Array', bound=np.ndarray)

def expm(A:Array) -> Array:
    if A.ndim < 2:
        raise LinAlgError('The input array must be at least two-dimensional')
    if A.shape[-1] != A.shape[-2]:
        raise LinAlgError('Last 2 dimensions of the array must be square')
    N = A.shape[0]
    if np.issubdtype(A.dtype,np.float32):
        return expm_float(A).reshape(N,N)
    elif np.issubdtype(A.dtype,np.float64):
        return expm_double(A).reshape(N,N)
    elif np.issubdtype(A.dtype,np.complex64):
        return expm_complex(A).reshape(N,N)
    elif np.issubdtype(A.dtype,np.complex128):
        return expm_doublecomplex(A).reshape(N,N)
    else:
        raise TypeError('Unsupported the Array type ' + str(A.dtype))
" > expm.py

echo init python package completed