import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set page configuration
st.set_page_config(page_title="Healthcare Facility Analysis in Nigeria", layout="wide")

# Title and Description
st.title("Healthcare Facility Analysis in Nigeria")
st.markdown("""
This app provides interactive geospatial analysis of healthcare facilities across selected locations in Nigeria. The analysis includes:
- Distribution of hospitals.
- Optimal travel route.
- Areas with poor hospital accessibility and potential for new hospitals.
- Statistical summaries of healthcare facilities.
Use the controls below to interact with the map and download the results in HTML format.
""")

# Data Sources and Update Information
DATA_SOURCES = """
### Data Sources
1. **Administrative Boundaries**: [GADM](https://gadm.org/download_country.html)
2. **Healthcare Facilities**: [Humanitarian Data Exchange](https://data.humdata.org/dataset/nigeria-health-care-facilities-in-nigeria)

**Last Updated**: June 2024
"""

# Function to check if shapefile components exist
def check_shapefile_exists(base_path, basename):
    extensions = ['.shp', '.shx', '.dbf', '.prj']
    missing_files = []
    for ext in extensions:
        file_path = os.path.join(base_path, basename + ext)
        if not os.path.isfile(file_path):
            missing_files.append(basename + ext)
    return missing_files

# Caching the data loading functions to optimize performance
@st.cache_data
def load_lga_data(shapefile_path):
    missing = check_shapefile_exists(os.path.dirname(shapefile_path), os.path.splitext(os.path.basename(shapefile_path))[0])
    if missing:
        st.error(f"Missing shapefile components: {', '.join(missing)}")
        st.stop()
    return gpd.read_file(shapefile_path)

@st.cache_data
def load_healthcare_data(csv_path):
    if not os.path.isfile(csv_path):
        st.error(f"Healthcare facilities CSV file not found at path: {csv_path}")
        st.stop()
    return pd.read_csv(csv_path)

@st.cache_data
def load_roads_data(shapefile_path):
    missing = check_shapefile_exists(os.path.dirname(shapefile_path), os.path.splitext(os.path.basename(shapefile_path))[0])
    if missing:
        st.error(f"Missing roads shapefile components: {', '.join(missing)}")
        st.stop()
    return gpd.read_file(shapefile_path)

@st.cache_resource
def load_road_network(state):
    try:
        # Load the road network graph for the selected state
        G = ox.graph_from_place(f"{state}, Nigeria", network_type='drive')
        return G
    except Exception as e:
        st.error(f"An error occurred while loading the road network: {e}")
        st.stop()

# Define paths to your data files
# Ensure these paths are correct relative to your Streamlit app's location
LGA_SHAPEFILE_PATH = "gadm41_NGA_2.shp"  # Update with your actual path
HEALTHCARE_CSV_PATH = "grid3_nga_-_health_facilities_-1.csv"  # Update with your actual path
ROADS_SHAPEFILE_PATH = "NGA_roads.shp"  # Update with your actual path

# Load datasets
with st.spinner("Loading Administrative Boundaries..."):
    lga_gdf = load_lga_data(LGA_SHAPEFILE_PATH)

with st.spinner("Loading Healthcare Facilities Data..."):
    healthcare_df = load_healthcare_data(HEALTHCARE_CSV_PATH)

with st.spinner("Loading Roads Data..."):
    roads_gdf = load_roads_data(ROADS_SHAPEFILE_PATH)

# Create tabs for each section including Statistical Insights and Data Sources
tab1, tab2, tab3, tab4 = st.tabs([
    "Hospital Distribution", 
    "Optimal Route", 
    "Hospital Accessibility",
    "Statistical Insights"
])

# ===========================
# Hospital Distribution Tab
# ===========================
with tab1:
    st.header("Hospital Distribution Across Selected Locations")
    
    # State and LGA selection
    state = st.selectbox("Select State", sorted(lga_gdf["NAME_1"].unique()))
    lga = st.selectbox(
        "Select Local Government Area", 
        sorted(lga_gdf[lga_gdf["NAME_1"] == state]["NAME_2"].unique())
    )

    # Filter healthcare facilities based on selection
    selected_facilities = healthcare_df[
        (healthcare_df['statename'] == state) & 
        (healthcare_df['lganame'] == lga)
    ]

    # Create Folium map centered around the selected facilities
    if not selected_facilities.empty:
        mean_lat = selected_facilities["latitude"].mean()
        mean_lon = selected_facilities["longitude"].mean()
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=10)
        marker_cluster = MarkerCluster().add_to(m)
        
        # Define marker colors based on functionality
        func_colors = {
            "Functional": "green", 
            "Partially Functional": "orange", 
            "Not Functional": "red", 
            "Unknown": "gray"
        }
        
        # Add markers to the map
        for _, row in selected_facilities.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=row['prmry_name'] or row['alt_name'],
                icon=folium.Icon(color=func_colors.get(row['func_stats'], "blue"))
            ).add_to(marker_cluster)
        
        folium_static(m)
        st.download_button(
            "Download HTML Map", 
            data=m.get_root().render(), 
            file_name="hospital_distribution.html"
        )
    else:
        st.warning("No healthcare facilities found for the selected State and LGA.")

# ===========================
# Optimal Route Tab
# ===========================
with tab2:
    st.header("Optimal Travel Route and Transportation Efficiency")
    
    # State and LGA selection for Optimal Route
    st.subheader("Select State and Local Government Area for Routing")
    route_state = st.selectbox("Select State for Route", sorted(lga_gdf["NAME_1"].unique()), key="route_state")
    route_lga = st.selectbox(
        "Select Local Government Area for Route", 
        sorted(lga_gdf[lga_gdf["NAME_1"] == route_state]["NAME_2"].unique()),
        key="route_lga"
    )
    
    # Filter healthcare facilities based on route selection
    route_selected_facilities = healthcare_df[
        (healthcare_df['statename'] == route_state) & 
        (healthcare_df['lganame'] == route_lga)
    ]
    
    if not route_selected_facilities.empty:
        # Dropdowns for selecting start and end locations from existing facilities
        st.subheader("Select Start and Destination Hospitals")
        start_location = st.selectbox(
            "Start Location",
            sorted(route_selected_facilities['prmry_name'].dropna().unique()),
            key="start_location"
        )
        end_location = st.selectbox(
            "Destination Location",
            sorted(route_selected_facilities['prmry_name'].dropna().unique()),
            key="end_location"
        )
        
        # Check if start and destination are the same
        if start_location == end_location:
            st.warning("Start Location and Destination Location cannot be the same. Please select different locations.")
        else:
            # Retrieve coordinates for selected hospitals
            start_coords = route_selected_facilities[route_selected_facilities['prmry_name'] == start_location][['latitude', 'longitude']].values[0]
            end_coords = route_selected_facilities[route_selected_facilities['prmry_name'] == end_location][['latitude', 'longitude']].values[0]
            
            # Load the road network graph (cached)
            with st.spinner("Loading road network graph..."):
                G = load_road_network(route_state)
            
            try:
                # Find the nearest nodes
                start_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
                dest_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])
                
                # Compute the shortest path
                route = nx.shortest_path(G, start_node, dest_node, weight='length')
                
                # Calculate total distance in kilometers
                total_length = nx.shortest_path_length(G, start_node, dest_node, weight='length') / 1000  # meters to km
                
                # Plot the route on a Folium map
                route_map = ox.plot_route_folium(G, route, 
                                                route_map=folium.Map(
                                                    location=[(start_coords[0] + end_coords[0])/2, (start_coords[1] + end_coords[1])/2], 
                                                    zoom_start=12
                                                ))
                folium_static(route_map)
                
                # Display the distance
                st.success(f"The optimal route distance is **{total_length:.2f} km**.")
                
                st.download_button(
                    "Download Route Map as HTML", 
                    data=route_map.get_root().render(), 
                    file_name="optimal_route.html"
                )
            except Exception as e:
                st.error(f"An error occurred while calculating the route: {e}")
    else:
        st.warning("No healthcare facilities found for the selected State and LGA in the Optimal Route tab.")

# ===========================
# Hospital Accessibility Tab
# ===========================
with tab3:
    st.header("Areas with Poor Hospital Accessibility")
    
    # State and LGA selection for Accessibility
    st.subheader("Select State and Local Government Area for Accessibility")
    acc_state = st.selectbox("Select State for Accessibility", sorted(lga_gdf["NAME_1"].unique()), key="acc_state")
    acc_lga = st.selectbox(
        "Select Local Government Area for Accessibility", 
        sorted(lga_gdf[lga_gdf["NAME_1"] == acc_state]["NAME_2"].unique()),
        key="acc_lga"
    )
    
    # Filter healthcare facilities based on accessibility selection
    acc_selected_facilities = healthcare_df[
        (healthcare_df['statename'] == acc_state) & 
        (healthcare_df['lganame'] == acc_lga)
    ]
    
    if not acc_selected_facilities.empty:
        # Buffer distance slider with explanation
        st.subheader("Configure Accessibility Buffer")
        st.markdown("""
        **Buffer Distance (km)**: This determines the radius around a central hospital. Areas within this radius are considered easily accessible, while areas outside may have poor access to healthcare facilities.
        """)
        buffer_distance = st.slider("Select Buffer Distance (km)", min_value=5, max_value=50, value=10)
        
        # Select a central hospital
        central_hospital = st.selectbox(
            "Select a Central Hospital", 
            sorted(acc_selected_facilities['prmry_name'].dropna().unique()),
            key="central_hospital"
        )
        
        # Get hospital coordinates
        hospital_row = acc_selected_facilities[acc_selected_facilities['prmry_name'] == central_hospital]
        if not hospital_row.empty:
            hospital_coords = hospital_row[['latitude', 'longitude']].values[0]
            hospital_point = Point(hospital_coords[1], hospital_coords[0])
            
            # Create buffer area using Shapely's buffer (approximated with degrees; for better accuracy, use geodesic buffers)
            # One degree of latitude is approximately 110.574 km
            buffer_degree = buffer_distance / 110.574
            buffer_area = hospital_point.buffer(buffer_degree)
            
            # Convert buffer_area to GeoDataFrame for plotting
            buffer_gdf = gpd.GeoDataFrame({'geometry': [buffer_area]}, crs="EPSG:4326")
            
            # Calculate distances to all other hospitals
            def calculate_distance(lat1, lon1, lat2, lon2):
                return ox.distance.great_circle_vec(lat1, lon1, lat2, lon2) / 1000  # meters to km
            
            acc_selected_facilities = acc_selected_facilities.copy()  # To avoid SettingWithCopyWarning
            acc_selected_facilities['distance_km'] = acc_selected_facilities.apply(
                lambda row: calculate_distance(hospital_coords[0], hospital_coords[1], row['latitude'], row['longitude']),
                axis=1
            )
            
            # Exclude the central hospital from the distance calculations
            other_hospitals = acc_selected_facilities[acc_selected_facilities['prmry_name'] != central_hospital]
            
            if not other_hospitals.empty:
                # Find the nearest and farthest hospitals
                nearest_hospital = other_hospitals.loc[other_hospitals['distance_km'].idxmin()]
                farthest_hospital = other_hospitals.loc[other_hospitals['distance_km'].idxmax()]
                
                # Display nearest and farthest hospital information
                st.markdown("### Nearest and Farthest Hospitals")
                col_near, col_far = st.columns(2)
                
                with col_near:
                    st.markdown(f"**Nearest Hospital:** {nearest_hospital['prmry_name']} ({nearest_hospital['distance_km']:.2f} km)")
                with col_far:
                    st.markdown(f"**Farthest Hospital:** {farthest_hospital['prmry_name']} ({farthest_hospital['distance_km']:.2f} km)")
            else:
                st.info("There are no other hospitals in the selected Local Government Area.")
            
            # Create Folium map
            m_buffer = folium.Map(location=[hospital_coords[0], hospital_coords[1]], zoom_start=12)
            
            # Add buffer polygon to the map
            folium.GeoJson(
                buffer_gdf,
                name="Buffer Area",
                style_function=lambda x: {
                    "fillColor": "#ff7800",
                    "color": "#ff7800",
                    "weight": 2,
                    "fillOpacity": 0.2,
                },
                tooltip=f"Buffer Area: {buffer_distance} km"
            ).add_to(m_buffer)
            
            # Add central hospital marker
            folium.Marker(
                location=[hospital_coords[0], hospital_coords[1]],
                popup=central_hospital,
                icon=folium.Icon(color='red', icon='plus')
            ).add_to(m_buffer)
            
            # Add other hospitals within buffer and mark nearest and farthest
            for _, row in acc_selected_facilities.iterrows():
                if row['prmry_name'] != central_hospital:
                    point = Point(row['longitude'], row['latitude'])
                    distance = row['distance_km']
                    if distance <= buffer_distance:
                        # Determine marker color
                        if row['prmry_name'] == nearest_hospital['prmry_name']:
                            icon_color = 'green'
                            icon_icon = 'star'
                            popup_text = f"{row['prmry_name']} (Nearest: {distance:.2f} km)"
                        elif row['prmry_name'] == farthest_hospital['prmry_name']:
                            icon_color = 'purple'
                            icon_icon = 'star'
                            popup_text = f"{row['prmry_name']} (Farthest: {distance:.2f} km)"
                        else:
                            icon_color = 'blue'
                            icon_icon = 'plus'
                            popup_text = row['prmry_name']
                        
                        folium.Marker(
                            location=[row['latitude'], row['longitude']],
                            popup=popup_text,
                            icon=folium.Icon(color=icon_color, icon=icon_icon)
                        ).add_to(m_buffer)
            
            folium_static(m_buffer)
            st.download_button(
                "Download Accessibility Map as HTML", 
                data=m_buffer.get_root().render(), 
                file_name="accessibility_map.html"
            )
        else:
            st.warning("Selected central hospital not found in the dataset.")
    else:
        st.warning("No healthcare facilities found for the selected State and LGA in the Hospital Accessibility tab.")

# ===========================
# Statistical Insights Tab
# ===========================
with tab4:
    st.header("Statistical Insights")
    
    st.subheader("Compare Multiple States and Local Governments")
    
    # Multi-select for States
    selected_states = st.multiselect(
        "Select States for Comparison",
        options=sorted(lga_gdf["NAME_1"].unique()),
        default=sorted(lga_gdf["NAME_1"].unique())  # Default to all states
    )
    
    # Filter LGAs based on selected states
    if selected_states:
        filtered_lgas = lga_gdf[lga_gdf["NAME_1"].isin(selected_states)]["NAME_2"].unique()
    else:
        filtered_lgas = []
    
    # Multi-select for LGAs
    selected_lgas = st.multiselect(
        "Select Local Government Areas for Comparison",
        options=sorted(filtered_lgas),
        default=sorted(filtered_lgas)  # Default to all LGAs in selected states
    )
    
    # Filter healthcare data based on selections
    if selected_states and selected_lgas:
        filtered_healthcare_df = healthcare_df[
            (healthcare_df['statename'].isin(selected_states)) &
            (healthcare_df['lganame'].isin(selected_lgas))
        ]
    elif selected_states:
        filtered_healthcare_df = healthcare_df[
            (healthcare_df['statename'].isin(selected_states))
        ]
    elif selected_lgas:
        filtered_healthcare_df = healthcare_df[
            (healthcare_df['lganame'].isin(selected_lgas))
        ]
    else:
        filtered_healthcare_df = healthcare_df.copy()
    
    # Check if filtered data is not empty
    if not filtered_healthcare_df.empty:
        # Number of hospitals per state
        hospitals_per_state = filtered_healthcare_df['statename'].value_counts().reset_index()
        hospitals_per_state.columns = ['State', 'Number of Hospitals']
        
        # Number of hospitals per LGA
        hospitals_per_lga = filtered_healthcare_df['lganame'].value_counts().reset_index()
        hospitals_per_lga.columns = ['Local Government Area', 'Number of Hospitals']
        
        # Functionality status distribution
        func_status = filtered_healthcare_df['func_stats'].value_counts().reset_index()
        func_status.columns = ['Functionality Status', 'Count']
        
        # Display the statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Number of Hospitals per State")
            fig1, ax1 = plt.subplots(figsize=(10, max(6, len(hospitals_per_state)*0.5)))
            sns.barplot(data=hospitals_per_state, x='Number of Hospitals', y='State', palette='viridis', ax=ax1)
            ax1.set_title("Hospitals Distribution by State")
            ax1.set_xlabel("Number of Hospitals")
            ax1.set_ylabel("State")
            st.pyplot(fig1)
        
        with col2:
            st.subheader("Number of Hospitals per Local Government Area")
            fig2, ax2 = plt.subplots(figsize=(10, max(6, len(hospitals_per_lga)*0.5)))
            sns.barplot(data=hospitals_per_lga, x='Number of Hospitals', y='Local Government Area', palette='magma', ax=ax2)
            ax2.set_title("Hospitals Distribution by LGA")
            ax2.set_xlabel("Number of Hospitals")
            ax2.set_ylabel("Local Government Area")
            st.pyplot(fig2)
        
        st.subheader("Functionality Status of Hospitals")
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.countplot(data=filtered_healthcare_df, 
                      y='func_stats', palette='coolwarm', ax=ax3)
        ax3.set_title("Functionality Status of Hospitals")
        ax3.set_xlabel("Count")
        ax3.set_ylabel("Functionality Status")
        st.pyplot(fig3)
        
        # Displaying the data in tables
        st.subheader("Data Tables")
        st.markdown("**Hospitals per State**")
        st.dataframe(hospitals_per_state)
        
        st.markdown("**Hospitals per Local Government Area**")
        st.dataframe(hospitals_per_lga)
        
        st.markdown("**Functionality Status Distribution**")
        st.dataframe(func_status)
        
    else:
        st.warning("No healthcare facilities found for the selected States and Local Government Areas.")

# Footer with Data Sources
st.markdown("---")
st.markdown(DATA_SOURCES)
st.markdown(f"**App Last Updated**: {datetime.now().strftime('%Y-%m-%d')}")
st.markdown("""
### Connect with Me
[Connect with me on LinkedIn](https://www.linkedin.com/in/adediran-adeyemi-17103b114/)
""")

