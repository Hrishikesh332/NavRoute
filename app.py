import streamlit as st
import heapq
import networkx as nx
import plotly.graph_objects as go


page_element="""
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://www.thewowstyle.com/wp-content/uploads/2021/03/How-to-Select-the-Ideal-Route-Optimization-Software.png");
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

with st.sidebar:
    st.title("NavRoute 🔍 - Help Navigating the Route")

    st.write("NavRoute is an application which helps to find the optimal route to traverse from one node to another node with the help of concept of Djikstra's Algorithm 👨‍💻")


    st.write('''The various features of NavRoute are:

🛣️ Visualizing the node and connection present in the network

🖧 Estimate the Shortest Route with the help of Djisktra's Algorithm''')



def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    heap = [(0, start)]
    
    while heap:
        (current_distance, current_node) = heapq.heappop(heap)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))
    
    return distances

st.title("Route Optimization with Dijkstra's Algorithm")

# st.write("In an interconnected graph, the Dijkstra's method is used to determine the shortest route between a specific node, known as the source node 📍, and every other node. The source node serves as the root of the shortest path tree that is created 🌳. It is often used in networks to create the most cost-effective paths possible.")

graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'D': 3},
    'C': {'D': 1, 'E': 5},
    'D': {'E': 1},
    'E': {}
}


# Create a NetworkX graph
G = nx.Graph()

# Add nodes and edges to the graph
for node in graph:
    G.add_node(node)
    for neighbor in graph[node]:
        G.add_edge(node, neighbor, weight=graph[node][neighbor])

# Get the positions of the nodes using the spring layout algorithm
pos = nx.spring_layout(G)

# Create a Plotly figure
fig = go.Figure()

# Add nodes to the figure
for node in G.nodes:
    x, y = pos[node]
    fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(size=20), name=f"Node {node}"))

# Add edges to the figure
for edge in G.edges:
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    weight = G.edges[edge]['weight']
    fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=weight), name='Edges'))

# Set the layout of the figure
# fig.update_layout(title='Interactive Network Plot for Graph', showlegend=False, hovermode='closest',  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
fig.update_layout(title='Interactive Network Plot for Graph', showlegend=True, hovermode='closest', plot_bgcolor='rgba(255, 255, 255, 0.7)', paper_bgcolor='rgba(255, 255, 255, 0.7)')
# Create the Streamlit app
st.title('Interactive Network Plot for Graph')

# Add a Plotly chart to the app
st.plotly_chart(fig, use_container_width=True)

start_node = st.selectbox("Select the start node", list(graph.keys()))
end_node = st.selectbox("Select the end node", list(graph.keys()))


distances = dijkstra(graph, start_node)
shortest_path = []
current_node = end_node

while current_node != start_node:
    shortest_path.append(current_node)
    for neighbor, weight in graph[current_node].items():
        if distances[neighbor] + weight == distances[current_node]:
            current_node = neighbor
            break

shortest_path.append(start_node)
shortest_path.reverse()

st.write(f"The shortest path from {start_node} to {end_node} is: {' -> '.join(shortest_path)}")


