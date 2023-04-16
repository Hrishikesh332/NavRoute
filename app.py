import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.graph_objs as go
import io
import json
import requests
from PIL import Image

def load_image(img):
    im=Image.open(img)
    return im
size=20



page_element="""
<style>
[data-testid="stAppViewContainer"]{
background-image: url("");
background-size: cover;
}
[data-testid="stHeader"]{
background-color: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
right: 2rem;
background-size: cover;
}
[data-testid="stSidebar"]> div:first-child{
background-image: url("https://img.freepik.com/free-vector/abstract-watercolor-pastel-background_87374-139.jpg");
background-size: cover;
}
</style>

"""
st.markdown(page_element, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;';>NavRoute 🔍</h1>", unsafe_allow_html=True)
st.markdown("---")


selected = option_menu(
            menu_title=None,  
            options=["Home","Route","About Us"],  
            icons=["home", "joystick", "reply-all-fill"],  
            menu_icon="cast",  
            default_index=0,  
            orientation="horizontal",
        )

def dijkstra(graph, start, end):

        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        previous_nodes = {node: None for node in graph}

        unvisited_nodes = set(graph.keys())

        while unvisited_nodes:
            current_node = min(unvisited_nodes, key=lambda node: distances[node])

            if distances[current_node] == float('inf'):
                break

            unvisited_nodes.remove(current_node)

            for neighbor, distance in graph[current_node].items():
                new_distance = distances[current_node] + distance
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node

        if previous_nodes[end] is None:
            return None

        path = []
        current_node = end
        while current_node is not None:
            path.append(current_node)
            current_node = previous_nodes[current_node]
        path.reverse()
        
        return path

with st.sidebar:
    st.image("logo.png")
    st.title("NavRoute 🔍 - Help Navigating the Route")

    st.write("NavRoute is an application which helps to find the optimal route to traverse from one node to another node with the help of concept of Djikstra's Algorithm 👨‍💻")


    st.write('''The various features of NavRoute are:

🛣️ Visualizing the node and connection present in the network

🖧 Estimate the Shortest Route with the help of Djisktra's Algorithm''')




if selected=="Home":
    st.subheader("Exploring the route with the help of NavRoute 🔍")

    col3, col4, col5, col6 = st.columns(4)
    with col3:
    
        st.markdown("![Alt Text](https://upload.wikimedia.org/wikipedia/commons/e/e4/DijkstraDemo.gif?w=200)")
    with col6:
        st.write("")
        st.write("")
        

        st.write("The objective of the NavRoute is to help you navigate to the shortest route with the help of Djiksta's Algorithm and also understand the path with network visualization 🌐")
    st.markdown("---")

    st.subheader("Working of Djikstra's Algorithm 🧑‍💼")
    col7, col8 = st.columns(2)
    with col8:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
    
        st.markdown("![Alt Text](https://realitybytesdotblog.files.wordpress.com/2017/07/path.jpg?w=600)")
    with col7:
        st.write("")
        st.write("")
    
        st.write('''
        
1. Create a set of unvisited nodes and set the distance of the starting node to 0 and all other nodes to infinity.

2. Select the node with the smallest distance and mark it as visited.

3. For each neighbor of the current node that has not been visited, calculate the distance to that node through the current node. If this distance is smaller than the current distance, update the distance.

4. Repeat steps 2 and 3 until all nodes have been visited or the destination node has been visited.

5. Once the destination node has been visited, the shortest path from the starting node to the destination node can be traced back by following the path with the smallest distance at each step.
''')
    st.markdown("---")


    graph = {
        'A': {'B': 2, 'C': 4},
        'B': {'D': 3},
        'C': {'D': 1, 'E': 5},
        'D': {'E': 1},
        'E': {}
    }

    st.subheader("Interactive Network Plot for Graph 📈")
    st.text("Demo:")
    st.code(
    '''
    graph = {
        'A': {'B': 2, 'C': 4},
        'B': {'D': 3},
        'C': {'D': 1, 'E': 5},
        'D': {'E': 1},
        'E': {}
    }
    
    ''')



    G = nx.Graph()
    for node in graph:
        G.add_node(node)
        for neighbor in graph[node]:
            G.add_edge(node, neighbor, weight=graph[node][neighbor])

    pos = nx.spring_layout(G)
    fig = go.Figure()

    for node in G.nodes:
        x, y = pos[node]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(size=20), name=f"Node {node}"))

    for edge in G.edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=weight), name='Edges'))

    fig.update_layout(title='', showlegend=True, hovermode='closest', plot_bgcolor='rgba(255, 255, 255, 0.7)', paper_bgcolor='rgba(255, 255, 255, 0.7)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")


    graph = {
        'A': {'B': 2, 'C': 4},
        'B': {'D': 3},
        'C': {'D': 1, 'E': 5},
        'D': {'E': 1},
        'E': {}
    }

    st.subheader('Shortest Path Finder')
    source_node = st.selectbox('Source node:', list(graph.keys()))
    destination_node = st.selectbox('Destination node:', list(graph.keys()))    


    if st.button('Find shortest path'):
        shortest_path = dijkstra(graph, source_node, destination_node)
        
        if shortest_path is None:
            st.write('No path found')
        else:
            st.write('Shortest path:', shortest_path)
            st.write('Total distance:', sum(graph[shortest_path[i]][shortest_path[i+1]] for i in range(len(shortest_path)-1)))




if selected == "Route":
    st.subheader("Instructions to follow:")
    st.code('''
    Instructions related to the file in which format, it is needed to be uplaoded:
    Source_Node       Destination_Node        Path_Cost

    ''')


    def read_excel_file(file):
        df = pd.read_excel(file, header=None, names=['Source', 'Target', 'Cost'])
        df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
        return df


    def create_network_graph(df):

        node_trace = go.Scatter(
            x=[], y=[], text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(color='rgb(255, 255, 255)',size=10, line=dict(width=2)))


        nodes = list(set(df['Source'].unique()) | set(df['Target'].unique()))
        for node in nodes:
            node_trace['x'] += [0]
            node_trace['y'] += [0]
            node_trace['text'] += [node]


        edge_trace = go.Scatter(
            x=tuple(df_edges['X']),
            y=tuple(df_edges['Y']),
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')
        edge_trace.update({'text': list(df_edges['cost'])})
        edge_trace.update({'hovertext': list(df_edges['cost'])})

        for index, row in df.iterrows():
            edge_trace['x'] += [nodes.index(row['Source'])+1, nodes.index(row['Target'])+1, None]
            edge_trace['y'] += [0, 0, None]

        layout = go.Layout(
            title='Network Graph',
            title_x=0.5,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))


        fig = go.Figure(data=[node_trace, edge_trace], layout=layout)

        return fig


    st.subheader("File Uploader and Network Graph:")
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])




    if uploaded_file is not None:
        df = read_excel_file(uploaded_file)
        df = df.drop(index=0)
        st.write(df)
        fig = create_network_graph(df)
        st.plotly_chart(fig)

    if uploaded_file is None:
        st.markdown("---")
        
        col3, col4, col5, col6, col7 = st.columns(5)
    
        with col5:
            st.subheader("OR")
        
        st.markdown("---")
        st.subheader("Custom Directory Path:")
        st.write('Enter graph as a dictionary:')
        graph_str = st.text_area('', "{'A': {'B': 2, 'C': 4}, 'B': {'D': 3}, 'C': {'D': 1, 'E': 5}, 'D': {'E': 1}, 'E': {}}")
        
        graph = eval(graph_str)
        source_node = st.selectbox('Source node:', list(graph.keys()))
        destination_node = st.selectbox('Destination node:', list(graph.keys()))


    if st.button('Find shortest path'):
        shortest_path = dijkstra(graph, source_node, destination_node)
        
        if shortest_path is None:
            st.write('No path found')
        else:
            st.write('Shortest path:', shortest_path)
            st.write('Total distance:', sum(graph[shortest_path[i]][shortest_path[i+1]] for i in range(len(shortest_path)-1)))
    
    
    
    
    
    
    G = nx.Graph()
    for node in graph:
        G.add_node(node)
        for neighbor in graph[node]:
            G.add_edge(node, neighbor, weight=graph[node][neighbor])

    pos = nx.spring_layout(G)

    fig = go.Figure()

    for node in G.nodes:
        x, y = pos[node]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(size=20), name=f"Node {node}"))

    for edge in G.edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=weight), name='Edges'))


    fig.update_layout(title='', showlegend=True, hovermode='closest', plot_bgcolor='rgba(255, 255, 255, 0.7)', paper_bgcolor='rgba(255, 255, 255, 0.7)')

    st.plotly_chart(fig, use_container_width=True)


if selected=="About Us":
    st.write("Made by Team: Network.AI")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Aditya Singh")
        # st.image("Aditya.jpg")

    with col2:

        st.header('Tanisha Shaikh')
        st.image("T.jPG")

    with col3:
        st.header("Ankita Sharma")
        st.image("Ankita.jpg")
    
