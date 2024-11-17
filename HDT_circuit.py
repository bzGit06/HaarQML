import pennylane as qml
from pennylane.operation import Operation


class Gen2QGate(Operation):
    num_wires = 2
    grad_method = None

    def __init__(self, weights, wires, id=None):
        super().__init__(weights, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(weights, wires):
        wires = qml.wires.Wires(wires)
        op_list = []

        op_list.append(qml.RZ(weights[0], wires=wires[0]))
        op_list.append(qml.RY(weights[1], wires=wires[0]))
        op_list.append(qml.RZ(weights[2], wires=wires[0]))

        op_list.append(qml.RZ(weights[3], wires=wires[1]))
        op_list.append(qml.RY(weights[4], wires=wires[1]))
        op_list.append(qml.RZ(weights[5], wires=wires[1]))

        op_list.append(qml.CNOT(wires=wires))

        op_list.append(qml.RY(weights[6], wires=wires[0]))
        op_list.append(qml.RZ(weights[7], wires=wires[1]))

        op_list.append(qml.CNOT(wires=wires[::-1]))

        op_list.append(qml.RY(weights[8], wires=wires[0]))

        op_list.append(qml.CNOT(wires=wires))

        op_list.append(qml.RZ(weights[9], wires=wires[0]))
        op_list.append(qml.RY(weights[10], wires=wires[0]))
        op_list.append(qml.RZ(weights[11], wires=wires[0]))

        op_list.append(qml.RZ(weights[12], wires=wires[1]))
        op_list.append(qml.RY(weights[13], wires=wires[1]))
        op_list.append(qml.RZ(weights[14], wires=wires[1]))

        return op_list


def one_qubit_circuit(params):
    Gen2QGate(weights=params[0], wires=[0, 1])
    Gen2QGate(weights=params[1], wires=[0, 2])
    Gen2QGate(weights=params[2], wires=[0, 3])

    return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))


def HEA_circuit(params_data, params_ancilla):
    paulis = [qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ]
    time = params_data.shape[0]
    depth = params_data.shape[1]
    time -= 2  # ignore data qubits
    for t in range(time):
        for l in range(depth):
            qml.RY(params_data[t, l, 0], wires=0)
            qml.RZ(params_data[t, l, 1], wires=0)
            qml.RY(params_data[t, l, 2], wires=1)
            qml.RZ(params_data[t, l, 3], wires=1)
            qml.RY(params_ancilla[t, l, 0], wires=t + 2)
            qml.RZ(params_ancilla[t, l, 1], wires=t + 2)

            qml.CNOT(wires=[0, t + 2])
            qml.CNOT(wires=[1, t + 2])

    return [qml.expval(p(0) @ q(1)) for p in paulis for q in paulis]


def HEA_circuit_bits(params_data, params_ancilla):
    time = params_data.shape[0]
    depth = params_data.shape[1]
    bath = params_ancilla.shape[0]
    for t in range(time):
        for l in range(depth):
            qml.RY(params_data[t, l, 0], wires=0)
            qml.RZ(params_data[t, l, 1], wires=0)
            qml.RY(params_data[t, l, 2], wires=1)
            qml.RZ(params_data[t, l, 3], wires=1)

            for b in range(bath):
                qml.RY(params_ancilla[b, t, l, 0], wires=2 + t * bath + b)
                qml.RZ(params_ancilla[b, t, l, 1], wires=2 + t * bath + b)

            qml.CNOT(wires=[0, 1])
            
            for b in range(bath - 1):
                qml.CNOT(wires=[2 + t * bath + b, 2 + t * bath + b + 1])

            qml.CNOT(wires=[1, 2 + t * bath])

    return qml.sample()
