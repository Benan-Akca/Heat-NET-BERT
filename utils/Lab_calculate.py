#%%
import numpy as np
import pandas as pd


''' 6028 V2318 ısıl işlem öncesi'''

# df = pd.read_excel('lab.xlsx')
# df.dropna(how="all",inplace=True,axis=1)
# df.dropna(how="all",inplace=True,axis=0)


# columns = ["vd_lambda","vd","sd_lambda","sd","ud_lambda","ud", "xyz_lambda", "x", "y", "z", "s_lambda", "s" ]

# df.columns = columns
# df = df.drop(0)
# df.to_pickle("C:\\Users\\benan\\OneDrive\\12_SISECAM\DATAFRAME\\df_lab_weight.pkl")
# # ww = df
# big_optic_df = pd.read_pickle(".\\lambda_dataset.pkl")
# optic_df = big_optic_df.iloc[0:445,:]
#%%


class SingleValues:

    @staticmethod
    def calculate_single_values(optic_df,pre_tempered=False):
        if pre_tempered == True:
            T_ = "T_pre"
            Rc_ = "Rc_pre"
            Ru_ = "Ru_pre"
        else:
            T_ = "T_post"
            Rc_ = "Rc_post"
            Ru_ = "Ru_post"
        ww = pd.read_pickle("C:\\Users\\benan\\OneDrive\\12_SISECAM\DATAFRAME\\df_lab_weight.pkl")

        U = 100 # spectral data sekmesindeki birim değişkeni
        #%% Visible single values
        vd_lambda = ww["vd_lambda"].dropna().values
        vd_optic =  optic_df[ optic_df["lambda"].isin(vd_lambda)]
        vd = ww[ww["vd_lambda"].isin(vd_lambda)].vd.values

        T = vd_optic[T_].values
        T_v = sum(vd * T) /100

        Rc = vd_optic[Rc_].values
        Rc_v = sum(vd * Rc) /100

        Ru = vd_optic[Ru_].values
        Ru_v = sum(vd * Ru) /100

        #%% Solar single values

        sd_lambda = ww["sd_lambda"].dropna().values
        sd_optic =  optic_df[ optic_df["lambda"].isin(sd_lambda)]
        sd = ww[ww["sd_lambda"].isin(sd_lambda)].sd.values

        T = sd_optic[T_].values
        T_s = sum(sd * T)

        Rc = sd_optic[Rc_].values
        Rc_s = sum(sd * Rc)

        Ru = sd_optic[Ru_].values
        Ru_s = sum(sd * Ru)

        #%% UV Single Value

        ud_lambda = ww["ud_lambda"].dropna().values
        ud_optic =  optic_df[ optic_df["lambda"].isin(ud_lambda)]
        ud = ww[ww["ud_lambda"].isin(ud_lambda)].ud.values

        T = ud_optic[T_].values
        T_u = sum(ud * T)

        #%% Colour calculations

        xyz_lambda = ww["xyz_lambda"].dropna().values
        xyz_optic =  optic_df[ optic_df["lambda"].isin(xyz_lambda)]

        T = xyz_optic[T_].values / U
        Rc = xyz_optic[Rc_].values / U
        Ru = xyz_optic[Ru_].values / U

        x = ww[ww["xyz_lambda"].isin(xyz_lambda)].x.values
        y = ww[ww["xyz_lambda"].isin(xyz_lambda)].y.values
        z = ww[ww["xyz_lambda"].isin(xyz_lambda)].z.values

        s = ww[ww["xyz_lambda"].isin(xyz_lambda)].s.values

        qx = sum(s*x)
        qy = sum(s*y)
        qz = sum(s*z)

        xT = sum(T*s*x) / qy * 100
        yT = sum(T*s*y) / qy * 100
        zT = sum(T*s*z) / qy * 100

        xRc = sum(Rc*s*x) / qy * 100
        yRc = sum(Rc*s*y) / qy * 100
        zRc = sum(Rc*s*z) / qy * 100

        xRu = sum(Ru*s*x) / qy * 100
        yRu = sum(Ru*s*y) / qy * 100
        zRu = sum(Ru*s*z) / qy * 100

        xxT = xT / (xT + yT + zT)
        yyT = yT / (xT + yT + zT)
        zzT = zT / (xT + yT + zT)

        xxRc = xRc / (xRc + yRc + zRc)
        yyRc = yRc / (xRc + yRc + zRc)
        zzRc = zRc / (xRc + yRc + zRc)

        xxRu = xRu / (xRu + yRu + zRu)
        yyRu = yRu / (xRu + yRu + zRu)
        zzRu = zRu / (xRu + yRu + zRu)

        #%% L a b calculation

        xn = qx / qy * 100
        yn = 100
        zn = qz / qy *100

        #L   -------------

        if yT / yn > 0.008856:
            LT = 116 * (yT / yn) ** (1 / 3) - 16
        else:
            LT = 903.3 * (yT / yn)

        if yRc / yn > 0.008856:
            LRc = 116 * (yRc / yn) ** (1 / 3) - 16
        else:
            LRc = 903.3 * (yRc / yn)

        if yRu / yn > 0.008856:
            LRu = 116 * (yRu / yn) ** (1 / 3) - 16
        else:
            LRu = 903.3 * (yRu / yn)

        # a* b* ----

        if xT / xn > 0.008856:
            fxT =  (xT / xn) ** (1 / 3) - 16 / 116 #???
        else:
            fxT = 7.787 * (xT / xn)

        if yT / yn > 0.008856:
            fyT =  (yT / yn) ** (1 / 3) - 16 / 116 #???
        else:
            fyT = 7.787 * (yT / yn)

        if zT / zn > 0.008856:
            fzT =  (zT / zn) ** (1 / 3) - 16 / 116 #???
        else:
            fzT = 7.787 * (zT / zn)

        aT = 500 * (fxT - fyT)
        bT = 200 * (fyT - fzT)

        #---

        if xRc / xn > 0.008856:
            fxRc =  (xRc / xn) ** (1 / 3) - 16 / 116 #???
        else:
            fxRc = 7.787 * (xRc / xn)

        if yRc / yn > 0.008856:
            fyRc =  (yRc / yn) ** (1 / 3) - 16 / 116 #???
        else:
            fyRc = 7.787 * (yRc / yn)

        if zRc / zn > 0.008856:
            fzRc =  (zRc / zn) ** (1 / 3) - 16 / 116 #???
        else:
            fzRc = 7.787 * (zRc / zn)

        aRc = 500 * (fxRc - fyRc)
        bRc = 200 * (fyRc - fzRc)

        #---

        if xRu / xn > 0.008856:
            fxRu =  (xRu / xn) ** (1 / 3) - 16 / 116 #???
        else:
            fxRu = 7.787 * (xRu / xn)

        if yRu / yn > 0.008856:
            fyRu =  (yRu / yn) ** (1 / 3) - 16 / 116 #???
        else:
            fyRu = 7.787 * (yRu / yn)

        if zRu / zn > 0.008856:
            fzRu =  (zRu / zn) ** (1 / 3) - 16 / 116 #???
        else:
            fzRu = 7.787 * (zRu / zn)

        aRu = 500 * (fxRu - fyRu)
        bRu = 200 * (fyRu - fzRu)


        columns = ["T_vis","Rc_vis", "Ru_vis", "T_solar", "Rc_solar", "Ru_solar", "T_uv","L_T", "L_Rc", "L_Ru", "a_T", "a_Rc", "a_Ru","b_T", "b_Rc", "b_Ru"]
        data = [[T_v, Rc_v, Ru_v, T_s, Rc_s, Ru_s, T_u, LT, LRc, LRu, aT, aRc, aRu,bT, bRc, bRu]]
        df_values = pd.DataFrame(data, columns = columns)
        return df_values
    @staticmethod
    def plot_lab(image_path,image_name,L_,a_,b_):
        build_gray = cv2.imread(image_path)
        L= L_ * 255 / 100
        a= a_ + 128
        b= b_ + 128
        lab_value = np.zeros((1,1,3))
        lab_value[:,:,0] = L
        lab_value[:,:,1] = a
        lab_value[:,:,2] = b
        lab_value = np.uint8(np.round(lab_value))
        bgr_value=cv2.cvtColor(lab_value,cv2.COLOR_Lab2BGR)
        build_gray = build_gray/255
        build_color=np.uint8(build_gray*bgr_value)
        build_color = cv2.resize(build_color,(400,266))
        cv2.imshow(image_name, build_color)
# %%
