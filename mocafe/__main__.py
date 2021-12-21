try:
    import mocafe
    import mocafe.angie
    import mocafe.fenut
    import mocafe.litforms
    import mocafe.expressions
    import mocafe.math

    print("Your mocafe is ready!")
except ModuleNotFoundError as e:

    print("There has been an issue")
    raise e
