import streamlit as st
import requests
from qiskit import QuantumCircuit
import plotly.graph_objects as go
import numpy as np
import time

st.set_page_config(page_title="Quantum Circuit Bloch Sphere Visualizer", layout="wide")
st.title("Quantum Circuit Bloch Sphere Visualizer")

# QASM input editor with default circuit
qasm_input = st.text_area("Edit Quantum Circuit QASM here:", height=200, value="""
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
""")

# # Noise probability slider
# noise_prob = st.slider("Depolarizing Noise Probability", min_value=0.0, max_value=0.5, value=0.0, step=0.01)

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

    vector = go.Scatter3d(
        x=[0, bx], y=[0, by], z=[0, bz],
        mode='lines+markers',
        line=dict(color='black', width=6),
        marker=dict(size=6, color='black'),
        name='Bloch Vector',
        hoverinfo='text',
        hovertext=f'({bx:.3f}, {by:.3f}, {bz:.3f})'
    )

    axis_lines = [
        go.Scatter3d(x=[-1.1,1.1], y=[0,0], z=[0,0], mode='lines', line=dict(color='white', width=4), name='X-axis'),
        go.Scatter3d(x=[0,0], y=[-1.1,1.1], z=[0,0], mode='lines', line=dict(color='lime', width=4), name='Y-axis'),
        go.Scatter3d(x=[0,0], y=[0,0], z=[-1.1,1.1], mode='lines', line=dict(color='lightblue', width=4), name='Z-axis')
    ]

    annotations = [
        dict(showarrow=False, x=1.1, y=0, z=0, text="X", font=dict(color="red", size=16)),
        dict(showarrow=False, x=0, y=1.1, z=0, text="Y", font=dict(color="green", size=16)),
        dict(showarrow=False, x=0, y=0, z=1.1, text="Z", font=dict(color="blue", size=16)),
    ]

    layout = go.Layout(
        scene=dict(
            xaxis=dict(range=[-1.2,1.2], title='X', autorange=False, showbackground=True, backgroundcolor="#e0f7fa"),
            yaxis=dict(range=[-1.2,1.2], title='Y', autorange=False, showbackground=True, backgroundcolor="#e0f7fa"),
            zaxis=dict(range=[-1.2,1.2], title='Z', autorange=False, showbackground=True, backgroundcolor="#e0f7fa"),
            aspectmode='cube',
            annotations=annotations
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=550,
        width=550,
        showlegend=True,
        title="3D Interactive Bloch Sphere Visualization"
    )

    fig = go.Figure(data=[sphere, vector] + axis_lines, layout=layout)
    return fig


# Button to run simulation
if st.button("Simulate & Visualize"):
    if not qasm_input.strip():
        st.error("Please enter a valid QASM circuit.")
    else:
        with st.spinner("Simulating quantum circuit..."):
            try:
                api_url = "http://127.0.0.1:5000/bloch"
                payload = {"qasm": qasm_input} 
                response = requests.post(api_url, json=payload)
                if response.status_code != 200:
                    st.error(f"Backend error: {response.text}")
                else:
                    data = response.json()
                    bloch_vectors = data.get("bloch_vectors", [])
                    animation_states = data.get("animation_states", [])
                    full_statevector = data.get("full_statevector", [])

                    # Display circuit diagram
                    try:
                        qc = QuantumCircuit.from_qasm_str(qasm_input)
                        st.subheader("Quantum Circuit Diagram")
                        st.pyplot(qc.draw(output='mpl', fold=120))
                    except Exception as e:
                        st.warning(f"Unable to draw circuit diagram: {e}")

                    st.markdown("---")
                    st.subheader("Bloch Sphere Visualizations")

                    n = len(bloch_vectors)
                    cols = st.columns(min(n, 4))

                    # Show Bloch vectors for each qubit
                    for idx, bv_dict in enumerate(bloch_vectors):
                        qubit = bv_dict["qubit"]
                        bv = bv_dict["bloch_vector"]
                        col = cols[idx % 4]
                        with col:
                            st.markdown(f"### Qubit {qubit}")
                            fig = plotly_bloch_sphere(bv)
                            st.plotly_chart(fig, use_container_width=True, key=f"bloch_{qubit}")

                            purity = sum(coord**2 for coord in bv)
                            st.write(f"Bloch Vector: (x={bv[0]:.3f}, y={bv[1]:.3f}, z={bv[2]:.3f})")
                            st.write(f"Purity: {purity:.3f} (1 means pure state)")

                    # Placeholder for gate animation
                    if animation_states:
                        st.markdown("---")
                        st.subheader("Gate Animation (Future Feature)")

                        st.info("Animation of gate effects will be implemented here.")

                        # Example: Show one frame animation with a slider (stub)
                        frame_index = st.slider("Animation Frame", 0, len(animation_states)-1, 0)
                        # You can visualize the frame's Bloch vectors similarly

                    else:
                        st.info("No animation states available from backend.")

            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
