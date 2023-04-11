from quadruped_reactive_walking import Params, KeyboardInput


p = Params.create_from_file()
kb = KeyboardInput(p)

kb.listen()
