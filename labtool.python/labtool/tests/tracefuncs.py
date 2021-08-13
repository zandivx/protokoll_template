import sys


def tracefunc(frame, event, arg, indent=[0]):
    if event == "call":
        indent[0] += 2
        print("-" * indent[0] + "> call function", frame.f_code.co_name)
    elif event == "return":
        print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
        indent[0] -= 2
    return tracefunc


def fn(frame, msg, arg):
    if msg != 'call':
        return
    # Filter as appropriate
    if frame.f_code.co_filename.startswith("/usr"):
        return
    print(f"Called {frame.f_code.co_name}")
    for i in range(frame.f_code.co_argcount):
        name = frame.f_code.co_varnames[i]
        print(f"\tArgument {name} = {frame.f_locals[name]}")
    return fn


def tracer(frame, event, arg):
    if event != "call":
        return None
    else:
        print(f"called function: {frame.f_code.co_name}")
        try:
            for i in range(frame.f_code.co_argcount):
                name = frame.f_code.co_varnames[i]
                print(f"\tArgument {name} = {frame.f_locals[name]}")
        except Exception as e:
            pass
            #print(f"\nEXCEPTION: {e}\n")
    return None


def tracer_2(frame, event, arg):
    indent = [0]

    def list_arguments():
        try:
            for i in range(frame.f_code.co_argcount):
                name = frame.f_code.co_varnames[i]
                print(f"\tArgument {name} = {frame.f_locals[name]}")
        except Exception as e:
            string = f"EXCEPTION: {e}"
            line = "\n" + "-" * len(string) + "\n"
            print(line + string + line)

    if event == "call":
        indent[0] += 2
        print("-" * indent[0] + "> call function", frame.f_code.co_name)
        list_arguments()
    elif event == "return":
        print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
        indent[0] -= 2
        list_arguments()


# sys.setprofile(tracer_2)
