from cffi import FFI

raw_model = b"""svm_type c_svc
kernel_type rbf
gamma 1
nr_class 2
total_sv 2
rho -2.90877
label 0 1
nr_sv 1 1
SV
256 0:0.3093766 1:0 2:0 3:0 4:0 5:0.1764706 6:0 7:0 8:1 9:0.1137485
-256 0:0.3332312 1:0 2:0 3:0 4:0.09657142 5:1 6:0 7:0 8:1 9:0.09917226
"""

ffi = FFI();
ffi.cdef("""
    int ffsvm_test(int);

    int ffsvm_context_create(void**);
    int ffsvm_set_max_problems(void*, unsigned int);
    int ffsvm_load_model(void*, char*);
    int ffsvm_predict_values(void*, float*, unsigned int, int*, unsigned int);
    int ffsvm_context_destroy(void**);
""");

num_features = 10
num_labels = 2

total_features = num_labels * num_features

# Will hold our context
ptr = ffi.new("void**");
model = ffi.new("char[]", raw_model);
features = ffi.new("float[]", [0] * total_features);
labels = ffi.new("int[]", [667] * num_labels);

# Features is flat array, just set data vector by vector
features = [
    0.3093766, 0.0, 0.0, 0.0, 0.0, 0.1764706, 0.0, 0.0, 1.0, 0.1137485,
    0.3332312, 0.0, 0.0, 0.0, 0.09657142, 1.0, 0.0, 0.0, 1.0, 0.09917226
]

# Load the lib
C = ffi.dlopen("../target/release/libffsvm.dylib");

# Some test calls
C.ffsvm_test(667);
C.ffsvm_context_create(ptr);
C.ffsvm_set_max_problems(ptr[0], 100);
C.ffsvm_load_model(ptr[0], model);
C.ffsvm_predict_values(ptr[0], features, total_features, labels, num_labels);
C.ffsvm_context_destroy(ptr);

# This is how things should be classified
assert labels[0] == 0
assert labels[1] == 1

print("FFI used successfully!")
