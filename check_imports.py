try:
    import nlopt
    print("NLopt: FOUND")
except ImportError:
    print("NLopt: MISSING")

try:
    import cupy
    print("CuPy: FOUND")
except ImportError:
    print("CuPy: MISSING")
