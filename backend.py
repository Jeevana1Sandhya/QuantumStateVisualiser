from flask import Flask, request, jsonify
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import DensityMatrix, partial_trace
import numpy as np

app = Flask(__name__)

def bloch_vector_from_reduced_dm(reduced_dm):
    pauli_x = np.array([[0,1], [1,0]])
    pauli_y = np.array([[0,-1j], [1j,0]])
    pauli_z = np.array([[1,0], [0,-1]])
    bx = np.real(np.trace(reduced_dm.data @ pauli_x))
    by = np.real(np.trace(reduced_dm.data @ pauli_y))
    bz = np.real(np.trace(reduced_dm.data @ pauli_z))
    return [bx, by, bz]

def complex_to_json(data):
    return [[c.real, c.imag] for c in data]

@app.route('/bloch', methods=['POST'])
def bloch():
    try:
        data = request.json
        qasm_str = data.get('qasm', '')
        if not qasm_str:
            return jsonify({'error': 'QASM string is required'}), 400

        qc = QuantumCircuit.from_qasm_str(qasm_str)
        num_qubits = qc.num_qubits

        sim = Aer.get_backend('aer_simulator')

        qc.save_statevector()
        result = sim.run(qc).result()
        data_result = result.data(0)

        if 'statevector' not in data_result:
            return jsonify({'error': f'Statevector not found in result. Keys: {list(data_result.keys())}'}), 500

        statevector = data_result['statevector']
        dm = DensityMatrix(statevector)
        statevector_json = complex_to_json(statevector)

        bloch_vectors = []
        for i in range(num_qubits):
            reduced = partial_trace(dm, [j for j in range(num_qubits) if j != i])
            bv = bloch_vector_from_reduced_dm(reduced)
            bloch_vectors.append({'qubit': i, 'bloch_vector': bv})

        animation_states = []  # Placeholder for future animation data

        response = {
            'bloch_vectors': bloch_vectors,
            'full_statevector': statevector_json,
            'animation_states': animation_states
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
