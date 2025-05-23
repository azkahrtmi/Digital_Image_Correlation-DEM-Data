from osgeo import gdal
import tempfile
import numpy as np
import cv2
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.colors as colors
import matplotlib.cm as cm
from scipy import signal as sg

def read_dem(uploaded_file, resolution):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    dataset = gdal.Open(tmp_path)   
    dataset = gdal.Warp("", dataset, xRes=resolution, yRes=resolution, resampleAlg="bilinear", format="MEM")
    dem = dataset.ReadAsArray().astype(np.float32)  # Konversi ke float32 agar kompatibel dengan OpenCV
    return dem

def save_plot_to_bytes(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer

def process_dem_files(dem1_file, dem2_file, resolution, temp_dim, b, d, olap):
    dem1 = read_dem(dem1_file, resolution)
    dem2 = read_dem(dem2_file, resolution)
    
    Dim_y, Dim_x = dem1.shape  
    
    olap_x = int((1 - olap) * (temp_dim + 2 * d))
    olap_y = int((1 - olap) * (temp_dim + 2 * b))
    
    h = int((Dim_y - (temp_dim + 2 * d)) / olap_y)
    w = int((Dim_x - (temp_dim + 2 * b)) / olap_x)
    
    print(f"Grid size: {h} rows x {w} cols")
    
    return dem1, dem2, h, w

def template_match(img_master, img_slave, method = 'cv2.TM_CCOEFF_NORMED', mlx = 1, mly = 1, show = True):
    """Melakukan pencocokan template antara dua gambar."""    
    
    img_master = cv2.resize(img_master,None,fx=mlx, fy=mly, interpolation = cv2.INTER_CUBIC)
    img_slave  = cv2.resize(img_slave,None,fx=mlx, fy=mly, interpolation = cv2.INTER_CUBIC)
    
    res = cv2.matchTemplate(img_slave,img_master,eval(method))

    w, h = img_master.shape[::-1]    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc 
    bottom_right = (top_left[0] + w, top_left[1] + h)

    px = (top_left[0]+bottom_right[0])/(2.0*mlx)
    py = (top_left[1]+bottom_right[1])/(2.0*mly)
    
    return px, py, max_val

def displacement(dem1, dem2, h, w, olap_x, olap_y, temp_dim, b, d, dim_pixel, frame_rate, mincor, mindef, maxdef):
    
    ml_x = 10
    ml_y = 10
    
    if int(temp_dim) % 2 == 0:
        temp_dim = temp_dim + 1
    
    results_mm = np.zeros((h * w, 6))
    k = 0
    
    img1r = dem1.copy()
    img2r = dem2.copy()
    
    for j in range(h):
        for i in range(w):
            

            Delta_X = i * olap_x
            Delta_Y = j * olap_y
            
            TP_temp_x = Delta_X+d+(temp_dim-1)/2.0 
            TP_temp_y = Delta_Y+b+(temp_dim-1)/2.0 
            
            start_x_template_slice = d+Delta_X
            stop_x_template_slice  = d+Delta_X+temp_dim
            start_y_template_slice = b+Delta_Y
            stop_y_template_slice  = b+Delta_Y+temp_dim
            
            shape_template = np.shape( img1r [  start_y_template_slice : stop_y_template_slice , start_x_template_slice : stop_x_template_slice ])
            
            assert np.allclose(shape_template[0], temp_dim)
            assert np.allclose(shape_template[1], temp_dim)
            assert np.allclose(stop_y_template_slice - start_y_template_slice, temp_dim)
            assert np.allclose(stop_x_template_slice - start_x_template_slice, temp_dim)
            assert np.allclose(TP_temp_x, (start_x_template_slice+ stop_x_template_slice)//2.0)
            assert np.allclose(TP_temp_y, (start_y_template_slice+ stop_y_template_slice)//2.0)
            
            start_x_search_slice =  Delta_X
            stop_x_search_slice  =  Delta_X + 2*d + temp_dim
            start_y_search_slice =  Delta_Y
            stop_y_search_slice  =  Delta_Y + 2*b + temp_dim
            
            shape_search = np.shape( img2r [  start_y_search_slice : stop_y_search_slice , start_x_search_slice : stop_x_search_slice ])
            
            assert np.allclose((shape_search[0] - temp_dim) /2.0, b)
            assert np.allclose((shape_search[1] - temp_dim) /2.0, d)
            assert np.allclose(TP_temp_x, (start_x_search_slice+ stop_x_search_slice)//2.0)
            assert np.allclose(TP_temp_y, (start_y_search_slice+ stop_y_search_slice)//2.0)
            
            temp =        img1r [  start_y_template_slice : stop_y_template_slice , start_x_template_slice : stop_x_template_slice ]
            search_area = img2r [  start_y_search_slice : stop_y_search_slice , start_x_search_slice : stop_x_search_slice ]
            
            indx,indy, maxcc = template_match(temp.astype('uint8'), search_area.astype('uint8'), mlx = ml_x, mly = ml_y, show = False)
            
            TP_search_x = Delta_X+indx - 0.5   
            TP_search_y = Delta_Y+indy - 0.5
            
            results_mm[k,0] = TP_temp_x     
            results_mm[k,1] = TP_temp_y     
            
            k2 = TP_search_x*dim_pixel-TP_temp_x*dim_pixel + results_mm[k,2] 
            k3 = TP_search_y*dim_pixel-TP_temp_y*dim_pixel + results_mm[k,3] 
            
            if maxcc >= mincor and np.sqrt((k3)**2 + (k2 )**2) * frame_rate > mindef and np.sqrt((k3)**2 + (k2 )**2) * frame_rate < maxdef  :
                        results_mm[k,2] = k2 
                        results_mm[k,3] = k3 
                        results_mm[k,4] = np.sqrt((results_mm[k,3])**2 + (results_mm[k,2] )**2)        
                        results_mm[k,5] = maxcc+results_mm[k,5]
            else:
                        results_mm[k,2] = 0
                        results_mm[k,3] = 0
                        results_mm[k,4] = 0
                        results_mm[k,5] = maxcc+results_mm[k,5]
                    

            k+=1
    
    dx = results_mm[:, 2].reshape(h, w)
    dy = results_mm[:, 3].reshape(h, w)
    
    fig_dx = plt.figure()
    plt.title("Displacement X")
    plt.imshow(dx, cmap=cm.jet, norm=mcolors.Normalize(vmin=np.min(dx), vmax=np.max(dx)))
    plt.colorbar(label='Displacement (m)')
    displacement_x_img = save_plot_to_bytes(fig_dx)
    plt.close(fig_dx)
    
    fig_dy = plt.figure()
    plt.title("Displacement Y")
    plt.imshow(dy, cmap=cm.jet, norm=mcolors.Normalize(vmin=np.min(dy), vmax=np.max(dy)))
    plt.colorbar(label='Displacement (m)')
    displacement_y_img = save_plot_to_bytes(fig_dy)
    plt.close(fig_dy)
  
    return dx, dy, displacement_x_img, displacement_y_img, results_mm

def strain (dx, dy, b, d, temp_dim, dim_pixel):
    operatore = np.array([[-1., 0, 1.]])
    gx = sg.convolve2d(dx, operatore / ((2 * (2 * d + temp_dim)) * dim_pixel), mode='same')
    gy = sg.convolve2d(dy, operatore.T / ((2 * (2 * b + temp_dim)) * dim_pixel), mode='same')
    
    fig_gx = plt.figure()
    plt.title("Strain X")
    plt.imshow(gx, cmap=cm.jet, norm=mcolors.Normalize(vmin=np.min(gx), vmax=np.max(gx)))
    plt.colorbar(label="Strain (µε)")
    strain_x_img = save_plot_to_bytes(fig_gx)
    plt.close(fig_gx)
    
    fig_gy = plt.figure()
    plt.title("Strain Y")
    plt.imshow(gy, cmap=cm.jet, norm=mcolors.Normalize(vmin=np.min(gy), vmax=np.max(gy)))
    plt.colorbar(label="Strain (µε)")
    strain_y_img = save_plot_to_bytes(fig_gy)
    plt.close(fig_gy)
    
    return gx, gy, strain_x_img, strain_y_img

def vektor (dem1, results_mm, frame_rate, maxcol, mincol, filename1, filename2):
    nz = mcolors.Normalize()
    nz.autoscale(results_mm[:,4]) 
    fig_vector = plt.figure()
    plt.title(f"Data {filename1} - {filename2}")
    ax = fig_vector.add_subplot(111)
    minv = np.min(dem1[dem1 != -32767])
    maxv = np.max(dem1)
    norm = colors.Normalize(minv, maxv)
    plt.imshow(dem1, cmap=plt.cm.gray, norm=norm)
    ax.set_prop_cycle('color',['red', 'black', 'yellow'])
           
    plt.gca().set_aspect('equal', adjustable='box')

    plt.ylabel('pixels')
           
    if maxcol<=0:
                maxcol=np.nanmax(results_mm[:,4])*frame_rate
    if mincol<=0:
                mincol=np.nanmin(results_mm[:,4])*frame_rate
            
    soglia_sup_prova_mm = maxcol
    soglia_inf_prova_mm = mincol + 0.001
                
            
    rcolbar = (np.nanmax(results_mm[:,4])-np.nanmin(results_mm[:,4]))*frame_rate/(maxcol-mincol) 
    plt.quiver( results_mm[:,0], 
                results_mm[:,1], 
                results_mm[:,2]/results_mm[:,4], 
                results_mm[:,3]/results_mm[:,4], 
                angles='xy', 
                scale=30,
                color=cm.jet(nz(results_mm[:,4]*rcolbar)),
                edgecolor='k', 
                linewidth=.2)

           
    cax,_ = mcolorbar.make_axes(plt.gca())
    cb = mcolorbar.ColorbarBase(cax, cmap=cm.jet, norm=mcolors.Normalize(vmin= soglia_inf_prova_mm, vmax= soglia_sup_prova_mm))
    cb.set_label('m/day')
    vector_img = save_plot_to_bytes(plt)
    
    return vector_img

def calculate_wind_direction(dx, dy):
    """
    Menghitung sudut dan mengklasifikasikan arah mata angin berdasarkan displacement X (dx) dan Y (dy).
    Sistem koordinat disesuaikan dengan sistem kartesian (dy dibalik).
    
    Args:
        dx (numpy array): Displacement X.
        dy (numpy array): Displacement Y.
    
    Returns:
        list: Daftar arah mata angin untuk setiap vektor.
    """
    # Konversi displacement menjadi sudut dalam derajat
    angle_rad = np.arctan2(-dy, dx)  # Membalik dy agar sesuai dengan sistem kartesian
    angle_deg = np.degrees(angle_rad)  # Konversi ke derajat
    angle_deg = (angle_deg + 360) % 360  # Pastikan nilai dalam rentang 0-360 derajat

    # Klasifikasi arah mata angin
    directions = []
    for angle in angle_deg:
        if (337.5 <= angle <= 360) or (0 <= angle < 22.5):
            directions.append("Timur")  # East (E)
        elif 22.5 <= angle < 67.5:
            directions.append("Timur Laut")  # Northeast (NE)
        elif 67.5 <= angle < 112.5:
            directions.append("Utara")  # North (N)
        elif 112.5 <= angle < 157.5:
            directions.append("Barat Laut")  # Northwest (NW)
        elif 157.5 <= angle < 202.5:
            directions.append("Barat")  # West (W)
        elif 202.5 <= angle < 247.5:
            directions.append("Barat Daya")  # Southwest (SW)
        elif 247.5 <= angle < 292.5:
            directions.append("Selatan")  # South (S)
        elif 292.5 <= angle < 337.5:
            directions.append("Tenggara")  # Southeast (SE)

    return directions, angle_deg



# def visualize_grid(dem1, dem2, h, w,filename1, filename2):
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))

#     for ax, dem, title in zip(axes, [dem1, dem2], [filename1, filename2]):
#         # Normalisasi nilai DEM agar sesuai dengan colormap
#         minv = np.min(dem[dem != -32767])  # Hindari nilai no-data
#         maxv = np.max(dem)
#         norm = colors.Normalize(minv, maxv)
        
#         # Plot DEM sebagai background dengan normalisasi
#         ax.imshow(dem, cmap=plt.cm.gray, norm=norm)

#         subset_height = dem.shape[0] / h
#         subset_width = dem.shape[1] / w

#         # Tambahkan grid
#         for i in range(1, w):
#             ax.vlines(i * subset_width, 0, dem.shape[0], colors='r', linestyles='dashed', linewidth=0.5)
#         for j in range(1, h):
#             ax.hlines(j * subset_height, 0, dem.shape[1], colors='r', linestyles='dashed', linewidth=0.5)

#         ax.set_title(title)
#         ax.set_xlabel("Kolom (X)")
#         ax.set_ylabel("Baris (Y)")
#         ax.set_aspect('equal')  # Pastikan aspek rasio sesuai dengan data

#     plt.tight_layout()
#     grid_img = save_plot_to_bytes(fig)
#     plt.close(fig)

#     return grid_img



# def template_matching(dem1, dem2, h, w, olap_x, olap_y, temp_dim):
#     correlation_results = []  # List untuk menyimpan hasil nilai korelasi

#     for i in range(h):
#         for j in range(w):
#             y_start = int(i * olap_y)
#             y_end = int(y_start + temp_dim)
#             x_start = int(j * olap_x)
#             x_end = int(x_start + temp_dim)
            
#             template = dem1[y_start:y_end, x_start:x_end]  # Ambil template dari DEM pertama
#             search_area = dem2[y_start:y_end, x_start:x_end]  # Area pencocokan di DEM kedua
            
#             # Panggil fungsi template_match
#             px, py, max_val = template_match(template, search_area)

#             correlation_results.append([i, j, px, py, max_val])  # Simpan hasil

#     # Konversi ke DataFrame untuk ditampilkan dalam tabel
#     correlation_df = pd.DataFrame(correlation_results, columns=["Grid Baris", "Grid Kolom", "X Match", "Y Match", "Nilai Korelasi"])
    
#     return correlation_df

