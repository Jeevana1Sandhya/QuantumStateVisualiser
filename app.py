import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import DensityMatrix, partial_trace, Statevector
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ---------- Helper Functions ----------

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

def run_quantum_simulation(qc, noisy=False):
    try:
        backend = Aer.get_backend("aer_simulator")
        if noisy:
            noise_model = NoiseModel()
            error = depolarizing_error(0.05, 1)  # 5% depolarizing error
            noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3", "cx"])
            result = backend.run(transpile(qc, backend), noise_model=noise_model).result()
        else:
            qc.save_statevector()
            result = backend.run(transpile(qc, backend)).result()

        statevector = result.get_statevector(qc, decimals=3)
        dm = DensityMatrix(statevector)

        num_qubits = qc.num_qubits
        results = []
        for i in range(num_qubits):
            reduced = partial_trace(dm, [j for j in range(num_qubits) if j != i])
            bv = bloch_vector_from_reduced_dm(reduced)
            rho = reduced.data / np.trace(reduced.data)
            purity = float(np.real(np.trace(rho @ rho)))
            length = float(np.linalg.norm(bv))
            results.append({
                "qubit": i,
                "bloch_vector": bv,
                "purity": purity,
                "length": length
            })
        return results, statevector, dm, None
    except Exception as e:
        return None, None, None, str(e)

def plotly_bloch_sphere(bloch_vector, purity, length):
    u, v = np.mgrid[0:2*np.pi:80j, 0:np.pi:40j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    sphere = go.Surface(x=x, y=y, z=z, opacity=0.3, showscale=False)

    bx, by, bz = bloch_vector
    data = [sphere]

    if length > 1e-6:
        vector = go.Scatter3d(x=[0,bx], y=[0,by], z=[0,bz],
                              mode="lines+markers",
                              line=dict(color="black", width=6),
                              marker=dict(size=6, color="black"))
        data.append(vector)
    else:
        dot = go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                           marker=dict(size=8, color="black"))
        data.append(dot)

    layout = go.Layout(
        scene=dict(xaxis=dict(range=[-1.2,1.2]), 
                   yaxis=dict(range=[-1.2,1.2]), 
                   zaxis=dict(range=[-1.2,1.2])),
        margin=dict(l=0,r=0,b=0,t=30), height=500, width=500
    )
    return go.Figure(data=data, layout=layout)

# ---------- Streamlit UI ----------

st.title("🌀 Quantum State Visualizer")

# Sidebar: Build circuit with gates
st.sidebar.header("⚡ Build Your Quantum Circuit")
num_qubits = st.sidebar.number_input("Number of qubits:", 1, 3, 1)
qc = QuantumCircuit(num_qubits)

# Choose gates interactively
gate = st.sidebar.selectbox("Add a gate:", ["None", "H", "X", "Y", "Z", "RX", "RY", "RZ", "CX"])
target = st.sidebar.number_input("Target qubit:", 0, num_qubits-1, 0)

if gate in ["RX", "RY", "RZ"]:
    angle = st.sidebar.slider("Rotation angle (radians)", 0.0, 2*np.pi, np.pi/2.0)

if gate == "CX" and num_qubits > 1:
    control = st.sidebar.number_input("Control qubit:", 0, num_qubits-1, 0)

if st.sidebar.button("Apply Gate"):
    if gate == "H": qc.h(target)
    elif gate == "X": qc.x(target)
    elif gate == "Y": qc.y(target)
    elif gate == "Z": qc.z(target)
    elif gate == "RX": qc.rx(angle, target)
    elif gate == "RY": qc.ry(angle, target)
    elif gate == "RZ": qc.rz(angle, target)
    elif gate == "CX" and num_qubits > 1: qc.cx(control, target)

st.subheader("Quantum Circuit Diagram")
st.pyplot(qc.draw(output="mpl"))

# Noise option
noisy = st.checkbox("Enable Noise (Depolarizing)", value=False)

if st.button("Simulate & Visualize"):
    with st.spinner("Running simulation..."):
        results, statevector, dm, error = run_quantum_simulation(qc, noisy=noisy)

    if error:
        st.error(f"Error: {error}")
    else:
        st.subheader("🧮 Statevector Representation")
        st.latex(Statevector(statevector).draw(output="latex"))

        st.subheader("🔮 Bloch Sphere Visualizations")
        for res in results:
            st.markdown(f"### Qubit {res['qubit']}")
            fig = plotly_bloch_sphere(res["bloch_vector"], res["purity"], res["length"])
            st.plotly_chart(fig, use_container_width=True)

            st.write(f"Purity = {res['purity']:.3f}, Length = {res['length']:.3f}")

        st.subheader("📊 Measurement Simulation (1000 shots)")
        qc.measure_all()
        backend = Aer.get_backend("aer_simulator")
        counts = backend.run(transpile(qc, backend), shots=1000).result().get_counts()
        st.pyplot(plot_histogram(counts).figure)

        # Export
        st.download_button("Download Circuit QASM", qc.qasm(), file_name="circuit.qasm")
