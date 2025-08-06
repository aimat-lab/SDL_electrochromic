import os
import json
import numpy as np
import matplotlib.pyplot as plt


def load(fn):
    data = []
    for lineidx,line in enumerate(open(fn, "r")):
        if lineidx>=9 and line!="\n":
            data.append([float(line.split(";")[0].replace(",",".")),
                        float(line.split(";")[1].replace(",",".")),
                        float(line.split(";")[2].replace(",",".")),
                        float(line.split(";")[3].replace(",",".")),
                        0.0])
    data = np.array(data)
    return(data)


fns = []
for fn in os.listdir("data/uv-vis"):
    if ".TXT" in fn:
        fns.append(fn)
fns = sorted(fns)

CIE_D65 = []
for lineidx, line in enumerate(open("references/CIE_std_illum_D65.csv")):
    CIE_D65.append([float(line.split(",")[0]), float(line.split(",")[1])])
CIE_D65 = np.array(CIE_D65)


CIE_xyz_10deg = []
for lineidx, line in enumerate(open("references/CIE_xyz_1964_10deg.csv")):
    if lineidx>0:
        line = line.replace("NaN", "0.0")
        CIE_xyz_10deg.append([float(line.split(",")[0]), float(line.split(",")[1]), float(line.split(",")[2]), float(line.split(",")[3])])
CIE_xyz_10deg = np.array(CIE_xyz_10deg)


CIE_D65_interp_x = CIE_xyz_10deg[:, 0]
CIE_D65_interp_y = np.interp(CIE_D65_interp_x, CIE_D65[:,0], CIE_D65[:,1])


plt.figure()
plt.plot(CIE_xyz_10deg[:, 0], CIE_xyz_10deg[:, 1], "-", color="C0", label="references/CIE xyz x")
plt.plot(CIE_xyz_10deg[:, 0], CIE_xyz_10deg[:, 2], "-", color="C1", label="references/CIE xyz y")
plt.plot(CIE_xyz_10deg[:, 0], CIE_xyz_10deg[:, 3], "-", color="C2", label="references/CIE xyz z")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Intensity [a.u.]")
plt.legend(loc="best")
plt.savefig("references/CIE_xyz_1964_10deg.png", dpi=120)
plt.close()


data_all = []
fns_used_unique = []
LAB_ref = {}
results = {}
filenames_shortened = []
for fn_idx, fn in enumerate(fns):
    data = load("data/uv-vis/%s"%(fn))        
    ID = fn.split("_")[1]
    data_all.append(data)
    filenames_shortened.append(ID)
    
    T = (data[:, 1] - data[:, 2])/(data[:, 3] - data[:, 2])*100.0
    A = -1.0*np.log10(0.01*T)
    idx_650 = np.argmin(np.abs(data[:, 0]-650.0))
    idx_900 = np.argmin(np.abs(data[:, 0]-900.0))
    idx_max = idx_650 + np.argmax(A[idx_650:idx_900])
    # idx_max = np.argmin(np.abs(data[:, 0]-700.0))
    lambda_max = data[idx_max, 0]
    A_max = A[idx_max]

    idx_380 = np.argmin(np.abs(data[:, 0]-380.0))
    idx_780 = np.argmin(np.abs(data[:, 0]-780.0))
    d_lambda = data[:, 0][1] - data[:, 0][0]
    D65_interp_x = data[:, 0][idx_380:idx_780]
    D65_interp_y = np.interp(D65_interp_x, CIE_D65[:,0], CIE_D65[:,1])
    tau_v = np.sum(T[idx_380:idx_780]*D65_interp_y*d_lambda)/np.sum(D65_interp_y*d_lambda)

    xyz_interp_x = data[:, 0][idx_380:idx_780]
    xyz_interp_y1 = np.interp(xyz_interp_x, CIE_xyz_10deg[:,0], CIE_xyz_10deg[:,1])
    xyz_interp_y2 = np.interp(xyz_interp_x, CIE_xyz_10deg[:,0], CIE_xyz_10deg[:,2])
    xyz_interp_y3 = np.interp(xyz_interp_x, CIE_xyz_10deg[:,0], CIE_xyz_10deg[:,3])
    N_y1 = np.sum(xyz_interp_y1*D65_interp_y*d_lambda)
    N_y2 = np.sum(xyz_interp_y2*D65_interp_y*d_lambda)
    N_y3 = np.sum(xyz_interp_y3*D65_interp_y*d_lambda)
    X = np.sum(T[idx_380:idx_780]*xyz_interp_y1*D65_interp_y*d_lambda)/N_y2
    Y = np.sum(T[idx_380:idx_780]*xyz_interp_y2*D65_interp_y*d_lambda)/N_y2
    Z = np.sum(T[idx_380:idx_780]*xyz_interp_y3*D65_interp_y*d_lambda)/N_y2
    X_w = 0.950489
    Y_w = 1.000
    Z_w = 1.088840
    x_r = 0.01*X / X_w
    y_r = 0.01*Y / Y_w
    z_r = 0.01*Z / Z_w
    epsilon = (24.0 / 116.0)**3.0
    kappa = 24389.0 / 27.0
    def f(t, epsilon, kappa):
        if t > epsilon:
            return(t**(1.0/3.0))
        else:
            return((kappa*t+16.0)/(116.0))
    L = 116.0 * f(y_r, epsilon, kappa) - 16.0
    a = 500.0 * (f(x_r, epsilon, kappa) - f(y_r, epsilon, kappa))
    b = 200.0 * (f(y_r, epsilon, kappa) - f(z_r, epsilon, kappa))

    print(fn)
    print("lambda_max: %.3f nm"%(lambda_max))
    print("A_max: %.3f"%(A_max))
    print("tau_v: %.3f %%"%(tau_v))
    print("X: %.3f"%(X))
    print("Y: %.3f"%(Y))
    print("Z: %.3f"%(Z))
    print("L: %.3f"%(L))
    print("a: %.3f"%(a))
    print("b: %.3f"%(b))
    results[fn] = [fn, lambda_max, A_max, tau_v, X, Y, Z, L, a, b]

    # plotting
    fig, axes = plt.subplots(3,1, sharex=True)
    axes[0].set_title(fn, fontsize=9)
    axes[0].plot(data[:, 0], data[:, 1], "-", color="C0", label="sample\n$\\lambda_{\\mathrm{max}} = %.0f \\mathrm{nm}$\n$A_{max} = %.3f$"%(lambda_max, A_max))
    axes[0].plot(data[:, 0], data[:, 2], "-", color="C1", label="dark")
    axes[0].plot(data[:, 0], data[:, 3], "-", color="C2", label="reference")
    axes[1].plot(data[:, 0], A, "-", color="C3", label="absorbance \nL* = %.1f\na* = %.1f\nb* = %.1f"%(L, a, b))
    axes[1].scatter([lambda_max], [A_max], facecolor="C3", edgecolor="k", s=40, zorder=10)
    axes[1].set_ylabel("Absorbance")
    axes[2].plot(data[:, 0], T, "-", color="C3", label="transmittance")
    axes[2].set_ylabel("Transmittance [%]")
    axes[0].legend(loc="best")
    axes[1].legend(loc="upper left")
    axes[2].legend(loc="best")
    axes[1].set_ylim([0, 1])
    axes[2].set_ylim([0, 1])
    axes[2].set_xlabel("Wavelength [nm]")
    axes[0].set_ylabel("Intensity [a.u.]")

    fns_used_unique.append(ID)
    plt.subplots_adjust(hspace=0)
    plt.savefig("analysis/uv-vis/%s.png"%(ID))
    plt.close()

cols = ['lambda_max', 'A_max', 'tau_v', 'X', 'Y', 'Z', 'L', 'a', 'b']
data_dict = {}
for fn in fns:
    res = results[fn] #ID, lambda_max, tau_v, X, Y, Z, L, a, b
    ID = res[0].split("_")[1].split("-")[1]
    values = res[1:]
    data_dict[ID] = {c: v for c, v in zip(cols, values)}

with open('./analysis/uv-vis/analysis_L_a_b_EC.json', 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, ensure_ascii=False, indent=4)
