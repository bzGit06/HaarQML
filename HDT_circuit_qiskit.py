from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit import QuantumCircuit, ParameterVector


def HEA_circuit_sampler(n_step, n_depth, n_data, n_bath):
    n_wires = n_data + n_bath
    n_param = 2 * n_depth * n_step
    n_register = n_data + n_bath * n_step

    # circuit parameters
    param_data = [ParameterVector(
        'data_' + str(d), length=n_param) for d in range(n_data)]
    param_bath = [ParameterVector(
        'bath_' + str(b), length=n_param) for b in range(n_bath)]

    q = QuantumRegister(n_wires, 'q')
    c = ClassicalRegister(n_register, 'c')
    circuit = QuantumCircuit(q, c)

    for t in range(n_step):
        for d in range(n_depth):
            param_id = 2 * d + 2 * n_depth * t
            for i in range(n_data):
                circuit.ry(param_data[i][param_id], i)
                circuit.rz(param_data[i][param_id + 1], i)

            for i in range(n_bath):
                circuit.ry(param_bath[i][param_id], n_data + i)
                circuit.rz(param_bath[i][param_id + 1], n_data + i)

            for i in range(n_wires // 2):
                circuit.cx(2 * i, 2 * i + 1)

            for i in range((n_wires - 1) // 2):
                circuit.cx(2 * i + 1, 2 * i + 2)

        register_id = n_data + t * n_bath
        circuit.measure(range(n_data, n_wires), range(
            register_id, register_id + n_bath))

        if t < n_step - 1:
            for b in range(n_bath):
                circuit.reset(n_data + b)

    circuit.measure(range(n_data), range(n_data))

    return circuit


def HEA_circuit_estimator(n_step, n_depth, n_data, n_bath):
    n_wires = n_data + n_bath
    n_param = 2 * n_depth * n_step

    # circuit parameters
    param_data = [ParameterVector(
        'data_' + str(d), length=n_param) for d in range(n_data)]
    param_bath = [ParameterVector(
        'bath_' + str(b), length=n_param) for b in range(n_bath)]

    q = QuantumRegister(n_wires, 'q')
    circuit = QuantumCircuit(q)

    for t in range(n_step):
        for d in range(n_depth):
            param_id = 2 * d + 2 * n_depth * t
            for i in range(n_data):
                circuit.ry(param_data[i][param_id], i)
                circuit.rz(param_data[i][param_id + 1], i)

            for i in range(n_bath):
                circuit.ry(param_bath[i][param_id], n_data + i)
                circuit.rz(param_bath[i][param_id + 1], n_data + i)

            for i in range(n_wires // 2):
                circuit.cx(2 * i, 2 * i + 1)

            for i in range((n_wires - 1) // 2):
                circuit.cx(2 * i + 1, 2 * i + 2)

        if t < n_step - 1:
            for b in range(n_bath):
                circuit.reset(n_data + b)

    return circuit
