import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import datetime
import zipfile
import dic  
# import io 

def create_zip(displacement_x_img, displacement_y_img, strain_x_img, strain_y_img, vector_img, dx, dy, gx, gy, df_csv):
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Simpan gambar dalam ZIP
        zip_file.writestr("displacement_x.png", displacement_x_img.getvalue())
        zip_file.writestr("displacement_y.png", displacement_y_img.getvalue())
        zip_file.writestr("strain_x.png", strain_x_img.getvalue())
        zip_file.writestr("strain_y.png", strain_y_img.getvalue())
        zip_file.writestr("vector.png", vector_img.getvalue())
       
       # Simpan hasil klasifikasi arah mata angin dalam format Excel
        excel_wind_buffer = BytesIO()
        with pd.ExcelWriter(excel_wind_buffer, engine='xlsxwriter') as writer:
            df_csv.to_excel(writer, sheet_name="Wind_Direction", index=False)
        excel_wind_buffer.seek(0)
        zip_file.writestr("wind_direction.xlsx", excel_wind_buffer.getvalue())


        # Simpan hasil perhitungan dalam file Excel
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            pd.DataFrame(dx).to_excel(writer, sheet_name="Displacement_X", index=False, header=False)
            pd.DataFrame(dy).to_excel(writer, sheet_name="Displacement_Y", index=False, header=False)
            pd.DataFrame(gx).to_excel(writer, sheet_name="Strain_X", index=False, header=False)
            pd.DataFrame(gy).to_excel(writer, sheet_name="Strain_Y", index=False, header=False)
        
        excel_buffer.seek(0)
        zip_file.writestr("results.xlsx", excel_buffer.getvalue())

    zip_buffer.seek(0)
    return zip_buffer




def main():
    st.title("Digital Image Correlation - DEM")
    
    dem1 = st.file_uploader("Upload DEM 1", type=["tif"])
    dem2 = st.file_uploader("Upload DEM 2", type=["tif"])
    
    st.sidebar.subheader("Configuration")
    resolution = st.sidebar.number_input("Resolution", value=1.0, step=0.1)
    dim_pixel = st.sidebar.number_input("Dimensi Pixel (m)", value=2.0, step=0.1)
    frame_rate = 1.0 / (st.sidebar.number_input("Camera acquisition time (jam)", value=24.0, step=1.0) / 24)
    templateWidth = st.sidebar.number_input("Template width [pixel]", value=64, step=1)
    b = st.sidebar.number_input("Edge y [pixel]", value=16, step=1)
    d = st.sidebar.number_input("Edge x [pixel]", value=16, step=1)
    olap = st.sidebar.slider("Overlap [%]", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
    mincor = st.sidebar.slider("Min Corr [0-0.99]", min_value=0.0, max_value=0.99, value=0.4, step=0.01)
    maxdef = st.sidebar.number_input("Max Def in m/day", value=1000.0, step=10.0)
    mindef = st.sidebar.number_input("Min Def in m/day", value=0.0, step=0.1)
    maxcol = st.sidebar.number_input("Max colbar in m/day", value=0.0, step=0.1)
    mincol = st.sidebar.number_input("Min colbar in m/day", value=0.0, step=0.1)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Results", "Visual", "Table", "Downloads"])
    
    if st.button("Run"):
        if not dem1 and not dem2:
            st.error("Anda belum mengunggah DEM 1 dan DEM 2!")
        elif not dem1:
            st.error("Anda belum mengunggah DEM 1!")
        elif not dem2:
            st.error("Anda belum mengunggah DEM 2!")
        else:
            st.success("Data Berhasil terupload")
            
            filename1 = dem1.name
            filename2 = dem2.name
            
        try:
            datetime1 = datetime.datetime(int(filename1[-14:-10]), int(filename1[-9:-7]), int(filename1[-6:-4]))
            datetime2 = datetime.datetime(int(filename2[-14:-10]), int(filename2[-9:-7]), int(filename2[-6:-4]))
            frame_rate = 1 / ((datetime2 - datetime1).total_seconds() / (60 * 60 * 24))
            
        except ValueError:
            st.error("Format nama file salah! Pastikan mengandung tanggal dalam format YYYYMMDD.")
            st.stop()
            
        msg = f" {1/frame_rate:.2f} hari"
            
        dem1_data, dem2_data, h, w = dic.process_dem_files(dem1, dem2, resolution, templateWidth, b, d, olap)
            
        dx, dy, displacement_x_img, displacement_y_img, results_mm = dic.displacement(
                dem1_data, dem2_data, h, w, 
                int((1 - olap) * (templateWidth + 2 * d)),
                int((1 - olap) * (templateWidth + 2 * b)), 
                templateWidth, b, d, dim_pixel, frame_rate, mincor, mindef, maxdef)           
            
        gx, gy, strain_x_img, strain_y_img = dic.strain(dx, dy, b, d, templateWidth, dim_pixel)

        vector_img = dic.vektor(dem1_data, results_mm, frame_rate, maxcol, mincol, filename1, filename2)
        
        directions, angles = dic.calculate_wind_direction(dx.flatten(), dy.flatten())

        # Simpan ke DataFrame (Data Lengkap)
        df_csv = pd.DataFrame({
            "X": results_mm[:, 0],
            "Y": results_mm[:, 1],
            "Sudut (°)": angles,
            "Arah Mata Angin": directions
        })

        # Hitung total jumlah vektor per arah mata angin untuk ditampilkan di Tab 3
        direction_counts = df_csv["Arah Mata Angin"].value_counts().reset_index()
        direction_counts.columns = ["Arah Mata Angin", "Jumlah"]

        with tab1:
                st.subheader("Results")
                st.markdown(f"""
                ### Informasi  
                **DEM 1:** {dem1.name}, **Ukuran:** {dem1_data.shape}  
                **DEM 2:** {dem2.name}, **Ukuran:** {dem2_data.shape}  
                **Jumlah Grid:** {h} Baris x {w} Kolom  
                **Total Hari:** {msg}  """)
                
        with tab2:
                st.subheader("Visual")
                st.image(displacement_x_img, caption="Displacement X", use_column_width=True)
                st.image(displacement_y_img, caption="Displacement Y", use_column_width=True)
                st.image(strain_x_img, caption="Strain X", use_column_width=True)
                st.image(strain_y_img, caption="Strain Y", use_column_width=True)
                st.image(vector_img, caption="Vector", use_column_width=True)

                
        with tab3:
                st.subheader("Table")
    #  
                data = {
                    "Metric": [
            "Mean Displacement X", "Max Displacement X", "Min Displacement X",
            "Mean Displacement Y", "Max Displacement Y", "Min Displacement Y",
            "Mean Strain x", "Max Strain x", "Min Strain x",
            "Mean Strain Y", "Max Strain Y", "Min Strain Y"
        ],
                      "Value (m or strain)": [
            np.mean(dx), np.max(dx), np.min(dx),
            np.mean(dy), np.max(dy), np.min(dy),
            np.mean(gx), np.max(gx), np.min(gx),
            np.mean(gy), np.max(gy), np.min(gy)
        ]
                }
                df = pd.DataFrame(data, index=np.arange(1, len(data["Metric"]) + 1)) 
                st.table(df)
                
                st.subheader("Total Jumlah Vektor per Arah Mata Angin")
                st.table(direction_counts)
                
        with tab4:
            st.subheader("Download semua Output")

            zip_buffer = create_zip(displacement_x_img, displacement_y_img, strain_x_img, strain_y_img, vector_img, dx, dy, gx, gy, df_csv)

            def extract_date(filename):
                try:
                    year = filename[-14:-10]
                    month = filename[-9:-7]
                    day = filename[-6:-4]
                    return f"{year}-{month}-{day}"
                except ValueError:
                    return "Unknown-Date"

            date1 = extract_date(filename1)
            date2 = extract_date(filename2)

            file_zip_name = f"DEM-{date1}_to_{date2}.zip"

            st.download_button(
                label="📥 Download ZIP",
                data=zip_buffer,
                file_name=file_zip_name,
                mime="application/zip"
            )
if __name__ == "__main__":
    main()