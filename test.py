from utils import *

R_BL = 0.031195905059576035
perf_BL = 39.06369801476684
CD_BL = 0.004852138459682465
cl = 0.65
best_perf=perf_BL
best_airfoil = None
try:
    log = np.loadtxt('results/log.txt')
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

while i < 1000:
    f = open('results/perf.log', 'a')
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
        af, R, a, b, perf, cd, CD_BL = lowestD(airfoil)
        if perf == np.nan:
            pass
        elif R < R_BL:
            mm = str(m).zfill(3)
            np.savetxt(f'BETTER/airfoil{mm}_{a}_{b}.dat', airfoil, header=f'airfoil{mm}_{a}_{b}', comments="")
            np.savetxt(f'BETTER/airfoil{mm}_{a}_{b}F.dat', af, header=f'airfoil{mm}_{a}_{b}F', comments="")
            f = open('results/perf.log', 'a')
            f.write(f'perf: {perf}, R: {R}, m: {mm}, a: {a}, b: {b}\n')
            f.close()
            m += 1
        k += 1
        log = np.array([i, k, m])
        np.savetxt('results/log.txt', log)
    k = 0
    i += 1