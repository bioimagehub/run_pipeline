import imagej
def test_imagej_1():
    # initialize ImageJ2 with Fiji plugins
    ij = imagej.init('sc.fiji:fiji')
    print(f"ImageJ2 version: {ij.getVersion()}")


def test_imagej_2():
    ij = imagej.init('sc.fiji:fiji')

    macro = """
#@ String name
#@ int age
#@ String city
#@output Object greeting
greeting = "Hello " + name + ". You are " + age + " years old, and live in " + city + "."
"""
    args = {
        'name': 'Chuckles',
        'age': 13,
        'city': 'Nowhere'
    }
    result = ij.py.run_macro(macro, args)
    print(result.getOutput('greeting'))

def test_imagej_3():
    ij = imagej.init('sc.fiji:fiji')

    ij.py.run_macro("""run("Blobs (25K)");""")
    blobs = ij.WindowManager.getCurrentImage()
    print(f"Blobs image dimensions: {blobs.getDimensions()}")
    ij.py.show(blobs)
    


# test_imagej_1()
# test_imagej_2()
test_imagej_3()