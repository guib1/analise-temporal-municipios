import threading

# Global lock to serialize access to HDF5/NetCDF libraries
# This prevents segfaults when multiple threads try to read HDF5 files simultaneously
HDF5_LOCK = threading.Lock()
