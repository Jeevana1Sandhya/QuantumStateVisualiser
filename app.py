import streamlit as st
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import DensityMatrix, partial_trace
import numpy as np
import plotly.graph_objects as go

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
            return None, None, None, f"Statevector not found in result keys: {list(data_result.keys())}"
        statevector = data_result['statevector']
        dm = DensityMatrix(statevector)
        statevector_json = complex_to_json(statevector)
        num_qubits = qc.num_qubits
        results = []
        for i in range(num_qubits):
            reduced = partial_trace(dm, [j for j in range(num_qubits) if j != i])
            bv = bloch_vector_from_reduced_dm(reduced)
            rho = reduced.data / np.trace(reduced.data)  # normalize
            purity = float(np.real(np.trace(rho @ rho)))
            length = float(np.linalg.norm(bv))
            results.append({
                'qubit': i,
                'bloch_vector': bv,
                'purity': purity,
                'length': length
            })
        return results, statevector_json, dm, None
    except Exception as e:
        return None, None, None, str(e)

def plotly_bloch_sphere(bloch_vector, purity, length):
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
    data = [sphere]

    if length > 1e-6:
        # Draw arrow if vector has non-zero length
        vector = go.Scatter3d(
            x=[0, bx], y=[0, by], z=[0, bz],
            mode='lines+markers',
            line=dict(color='black', width=6),
            marker=dict(size=6, color='black'),
            name='Bloch Vector',
            hoverinfo='text',
            hovertext=f'({bx:.3f}, {by:.3f}, {bz:.3f})'
        )
        data.append(vector)
    else:
        # Draw dot at origin for maximally mixed state
        dot = go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=8, color='black', symbol='circle'),
            name='Mixed State',
            hoverinfo='text',
            hovertext=f'Maximally Mixed (Purity={purity:.3f})'
        )
        data.append(dot)

    axis_lines = [
        go.Scatter3d(x=[-1.1,1.1], y=[0,0], z=[0,0], mode='lines', line=dict(color='red', width=4), name='X-axis'),
        go.Scatter3d(x=[0,0], y=[-1.1,1.1], z=[0,0], mode='lines', line=dict(color='green', width=4), name='Y-axis'),
        go.Scatter3d(x=[0,0], y=[0,0], z=[-1.1,1.1], mode='lines', line=dict(color='blue', width=4), name='Z-axis')
    ]
    annotations = [
        dict(showarrow=False, x=1.1, y=0, z=0, text="X", font=dict(color="red", size=16)),
        dict(showarrow=False, x=0, y=1.1, z=0, text="Y", font=dict(color="green", size=16)),
        dict(showarrow=False, x=0, y=0, z=1.1, text="Z", font=dict(color="blue", size=16)),
    ]
    layout = go.Layout(
        scene=dict(
            xaxis=dict(range=[-1.2,1.2], showbackground=True, backgroundcolor="#e0f7fa"),
            yaxis=dict(range=[-1.2,1.2], showbackground=True, backgroundcolor="#e0f7fa"),
            zaxis=dict(range=[-1.2,1.2], showbackground=True, backgroundcolor="#e0f7fa"),
            aspectmode='cube',
            annotations=annotations
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=500,
        width=500,
        showlegend=False,
        title="Bloch Sphere"
    )
    fig = go.Figure(data=data + axis_lines, layout=layout)
    return fig

# ---------------- Streamlit App ----------------

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
            results, full_statevector, dm, error = run_quantum_simulation(qasm_input)
            if error:
                st.error(f"Simulation error: {error}")
            else:
                try:
                    qc = QuantumCircuit.from_qasm_str(qasm_input)
                    st.subheader("Quantum Circuit Diagram")
                    st.pyplot(qc.draw(output='mpl', fold=120))
                except Exception as e:
                    st.warning(f"Unable to draw circuit diagram: {e}")

                st.markdown("---")
                st.subheader("Bloch Sphere Visualizations")

                for res in results:
                    qubit = res["qubit"]
                    bv = res["bloch_vector"]
                    purity = res["purity"]
                    length = res["length"]

                    st.markdown(f"### Qubit {qubit}")
                    fig = plotly_bloch_sphere(bv, purity, length)
                    st.plotly_chart(fig, use_container_width=True, key=f"bloch_{qubit}")

                    st.write(f"Bloch Vector: (x={bv[0]:.3f}, y={bv[1]:.3f}, z={bv[2]:.3f})")
                    st.write(f"Bloch Vector Length: {length:.3f}")
                    st.write(f"Purity: {purity:.3f} (1 = pure, {1/(2**1):.3f} = maximally mixed for 1 qubit)")

                st.markdown("---")
                st.subheader("Full Statevector (Real, Imaginary)")
                st.write(full_statevector)
