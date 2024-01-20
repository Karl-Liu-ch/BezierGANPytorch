import numpy as np
from xfoil import XFoil
from xfoil.model import Airfoil
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import logging
logging.basicConfig(filename='results/perf.log', encoding='utf-8', level=logging.DEBUG)
from utils import *
import gc

def evaluate(airfoil, cl = 0.65, Re1 = 5.8e4, Re2 = 4e5, lamda = 5, return_CL_CD=False, check_thickness = True):
        
    if detect_intersect(airfoil):
        # print('Unsuccessful: Self-intersecting!')
        perf = np.nan
        R = np.nan
        CD = np.nan
        
    elif (cal_thickness(airfoil) < 0.06 or cal_thickness(airfoil) > 0.09) and check_thickness:
        # print('Unsuccessful: Too thin!')
        perf = np.nan
        R = np.nan
        CD = np.nan
    
    elif np.abs(airfoil[0,0]-airfoil[-1,0]) > 0.01 or np.abs(airfoil[0,1]-airfoil[-1,1]) > 0.01:
        # print('Unsuccessful:', (airfoil[0,0],airfoil[-1,0]), (airfoil[0,1],airfoil[-1,1]))
        perf = np.nan
        R = np.nan
        CD = np.nan
        
    else:
        af = np.copy(airfoil)
        airfoil = setupflap(airfoil, theta=-2)
        airfoil = interpolate(airfoil, 300, 3)
        CD, _ = evalpreset(airfoil, Re=Re2)
        i = 0
        while CD < 0.004 and (not np.isnan(CD)) and i < 2:
            i += 1
            print(not np.isnan(CD), CD)
            airfoil = interpolate(airfoil, 200 + i * 100, 3)
            CD, _ = evalpreset(airfoil, Re=Re2 + i * 100)
            print(CD)
        if i >= 2:
            CD = np.nan
            
        airfoil = setflap(airfoil, theta=2)
        perf, _, cd = evalperf(airfoil, cl = cl, Re = Re1)
        R = cd + CD * lamda
        if R < 0.015 + 0.004852138459682465 * lamda and perf < 37:
            print(f'R: {R}, perf: {perf}, change reynolds')
            re2 = Re2 + 1000
            perf, CD, airfoil, R = evaluate(af, cl = 0.65, Re1 = 5.8e4, Re2 = re2, lamda = 3, return_CL_CD=False, check_thickness = True)
        if perf < -100 or perf > 300 or cd < 1e-3:
            perf = np.nan
        elif not np.isnan(perf):
            print('Successful: CL/CD={:.4f}, R={}'.format(perf, R))
            
    if return_CL_CD:
        return perf, cl.max(), cd.min()
    else:
        return perf, CD, airfoil, R

if __name__ == "__main__":
    R_BL = 0.031195905059576035
    perf_BL = 39.06369801476684
    CD_BL = 0.004852138459682465
    cl = 0.65
    best_perf=perf_BL
    best_airfoil = None
    try:
        log = np.loadtxt('results/simlog.txt')
        i = int(log[0])
        k = int(log[1])
        m = int(log[2])
    except:
        m = 0
        i = 0
        k = 0
    name = 'Airfoils'
    airfoilpath = '/work3/s212645/BezierGANPytorch/'+name+'/'

    print(f'i: {i}, k: {k}, m: {m}')
    while i < 100:
        f = open('results/simperf.log', 'a')
        f.write(f'files: {i}\n')
        f.close()
        num = str(i).zfill(3)
        airfoils = np.load(airfoilpath+num+'.npy')
        airfoils = delete_intersect(airfoils)
        while k < airfoils.shape[0]:
            airfoil = airfoils[k,:,:]
            airfoil = derotate(airfoil)
            airfoil = Normalize(airfoil)
            xhat, yhat = savgol_filter((airfoil[:,0], airfoil[:,1]), 10, 3)
            airfoil[:,0] = xhat
            airfoil[:,1] = yhat
            perf, CD, af, R = evaluate(airfoil, cl)
            if perf == np.nan:
                pass
            else:
                if R < R_BL:
                    mm = str(m).zfill(3)
                    np.savetxt(f'BETTER/airfoil{mm}.dat', airfoil, header=f'airfoil{mm}', comments="")
                    np.savetxt(f'BETTER/airfoil{mm}F.dat', af, header=f'airfoil{mm}F', comments="")
                    f = open('results/simperf.log', 'a')
                    f.write(f'perf: {perf}, R: {R}, better m: {mm}, i: {i}, k: {k}\n')
                    f.close()
                    m += 1
                if perf > perf_BL:
                    mm = str(m).zfill(3)
                    np.savetxt(f'samples/airfoil{mm}.dat', airfoil, header=f'airfoil{mm}', comments="")
                    np.savetxt(f'samples/airfoil{mm}F.dat', af, header=f'airfoil{mm}F', comments="")
                    f = open('results/simperf.log', 'a')
                    f.write(f'perf: {perf}, R: {R}, better m: {mm}, i: {i}, k: {k}\n')
                    f.close()
                    m += 1
                del f
            k += 1
            log = np.array([i, k, m])
            np.savetxt('results/simlog.txt', log)
            del airfoil
            gc.collect()
        k = 0
        i += 1
        del airfoils
        del f
        gc.collect()