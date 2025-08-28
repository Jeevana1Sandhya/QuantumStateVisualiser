import streamlit as st
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import DensityMatrix, partial_trace
import numpy as np
import plotly.graph_objects as go
from qiskit.visualization import plot_bloch_vector
import matplotlib.pyplot as plt

# --------- Helper functions ---------
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

def run_quantum_simulation(qasm_str):
    try:
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        sim = Aer.get_backend('aer_simulator')
        qc.save_statevector()
        result = sim.run(qc).result()
        data_result = result.data(0)
        if 'statevector' not in data_result:
            return None, None, f"Statevector not found in result keys: {list(data_result.keys())}"
        statevector = data_result['statevector']
        dm = DensityMatrix(statevector)
        statevector_json = complex_to_json(statevector)
        num_qubits = qc.num_qubits
        bloch_vectors = []
        for i in range(num_qubits):
            reduced = partial_trace(dm, [j for j in range(num_qubits) if j != i])
            bv = bloch_vector_from_reduced_dm(reduced)
            bloch_vectors.append({'qubit': i, 'bloch_vector': bv})
        return bloch_vectors, statevector_json, None
    except Exception as e:
        return None, None, str(e)

def plotly_bloch_sphere(bloch_vector):
    u, v = np.mgrid[0:2*np.pi:80j, 0:np.pi:40j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    sphere = go.Surface(
        x=x, y=y, z=z, opacity=0.3,
        colorscale=[[0, '#a6cee3'], [1, '#1f78b4']],
        showscale=False,
        lighting=dict(ambient=0.7, diffuse=0.7, roughness=0.6),
        hoverinfo='skip'
    )
    bx, by, bz = bloch_vector
    vector_line = go.Scatter3d(
        x=[0, bx], y=[0, by], z=[0, bz],
        mode='lines+markers',
        line=dict(color='black', width=6),
        marker=dict(size=6, color='black'),
        name='Bloch Vector'
    )
    axis_lines = [
        go.Scatter3d(x=[-1.2,1.2], y=[0,0], z=[0,0], mode='lines', line=dict(color='red', width=4), name='X-axis'),
        go.Scatter3d(x=[0,0], y=[-1.2,1.2], z=[0,0], mode='lines', line=dict(color='green', width=4), name='Y-axis'),
        go.Scatter3d(x=[0,0], y=[0,0], z=[-1.2,1.2], mode='lines', line=dict(color='blue', width=4), name='Z-axis')
    ]
    annotations = [
        dict(showarrow=False, x=1.25, y=0, z=0, text="X", font=dict(color="red", size=16)),
        dict(showarrow=False, x=0, y=1.25, z=0, text="Y", font=dict(color="green", size=16)),
        dict(showarrow=False, x=0, y=0, z=1.25, text="Z", font=dict(color="blue", size=16)),
    ]
    layout = go.Layout(
        scene=dict(
            xaxis=dict(range=[-1.3,1.3], showbackground=True, backgroundcolor="#e0f7fa"),
            yaxis=dict(range=[-1.3,1.3], showbackground=True, backgroundcolor="#e0f7fa"),
            zaxis=dict(range=[-1.3,1.3], showbackground=True, backgroundcolor="#e0f7fa"),
            aspectmode='cube',
            annotations=annotations
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=500,
        width=500,
        showlegend=False,
        title="3D Bloch Sphere"
    )
    fig = go.Figure(data=[sphere, vector_line] + axis_lines, layout=layout)
    return fig

# --------- Streamlit UI ---------
st.title("Quantum Circuit Bloch Sphere Visualizer")

qasm_input = st.text_area("Edit Quantum Circuit QASM here:", height=200, value="""
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
""")

if st.button("Simulate & Visualize"):
    if not qasm_input.strip():
        st.error("Please enter a valid QASM circuit.")
    else:
        with st.spinner("Simulating quantum circuit..."):
            bloch_vectors, full_statevector, error = run_quantum_simulation(qasm_input)
            if error:
                st.error(f"Simulation error: {error}")
            else:
                # Circuit Diagram
                try:
                    qc = QuantumCircuit.from_qasm_str(qasm_input)
                    st.subheader("Quantum Circuit Diagram")
                    st.pyplot(qc.draw(output='mpl', fold=120))
                except Exception as e:
                    st.warning(f"Unable to draw circuit diagram: {e}")

                st.markdown("---")
                st.subheader("Bloch Sphere Visualizations (3D + 2D)")

                for idx, bv_dict in enumerate(bloch_vectors):
                    qubit = bv_dict["qubit"]
                    bv = bv_dict["bloch_vector"]
                    st.markdown(f"### Qubit {qubit}")

                    # Side-by-side columns
                    col1, col2 = st.columns(2)

                    # 3D Bloch sphere
                    with col1:
                        st.markdown("**3D Interactive Bloch Sphere**")
                        fig3d = plotly_bloch_sphere(bv)
                        st.plotly_chart(fig3d, use_container_width=True, key=f"bloch3d_{qubit}")

                    # 2D Bloch sphere
                    with col2:
                        st.markdown("**2D Bloch Sphere**")
                        fig2d, ax = plt.subplots(figsize=(4,4))
                        plot_bloch_vector(bv, ax=ax)
                        st.pyplot(fig2d)
                        plt.close(fig2d)

                    # Purity info
                    purity = sum(coord**2 for coord in bv)
                    st.write(f"Bloch Vector: (x={bv[0]:.3f}, y={bv[1]:.3f}, z={bv[2]:.3f})")
                    st.write(f"Purity: {purity:.3f} (1 means pure state)")

                st.markdown("---")
                st.subheader("Full Statevector (Real, Imaginary)")
                st.write(full_statevector)
