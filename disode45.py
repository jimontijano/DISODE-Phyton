import numpy as np
import sys
from struct import Struct
from types import SimpleNamespace

def disode45(FUN, switchfun, tspan, Y, options=None):
    """

   DISODE45  is a program for the numerical integration of ODEs with
   DISCONTINUITIES.  Filippov type systems are allowed.
   It is based on the CMR54D pair of embedded Runge-Kutta methods specially 
   designed by M. Calvo, J.I. Montijano and L. Rández for this class of 
   problems
   The function or functions that define the manifolds of the
   discontinuities g(x,y) are supposed to be known, provided
   by the user.

  [TOUT,YOUT] = DISODE45(ODEFUN,SWITCHFUN, TSPAN,Y0), with TSPAN = [T0 TFINAL]
     integrates the system of differential equations y' = f(t,y) from time T0 
     to TFINAL with initial conditions Y0.
     
  Input arguments:
    ODEFUN is a function handle. For a scalar T and a vector Y, ODEFUN(T,Y)
         must return a column vector corresponding to f(t,y).
    SWITCHFUN is a function handle. [values, isterminal, direction]=SWITCHFUN(T,Y) must return
         a column vector VALUES in which the component i contains the value of the
         i function g_i(t,y) defining the i-eme discontinuity manifold.
         It also returns a column vector ISTERMINAL and a column vector
         DIRECTION.
  Output arguments:
    Each row in the solution array YOUT corresponds to a time
     returned in the column vector TOUT.

    - options: Objeto con las opciones de configuración (clase Struct)
  
    """
    
    Y = np.array(Y, dtype=float).flatten()
    
 # 1. Configuration of the de defaul values if they are nor provided in 'options'
 
    if options is None:
       options = SimpleNamespace()
    if not hasattr(options, 'AbsTol'): options.AbsTol = 1e-4
    if not hasattr(options, 'RelTol'): options.RelTol = 1e-4
    if not hasattr(options, 'Gradient'): options.Gradient = None
    if not hasattr(options, 'ActionSwitch'): options.ActionSwitch = None
    if not hasattr(options, 'GradientComponents'): options.GradientComponents=np.array([])
    if not hasattr(options, 'InitialStep'): options.InitialStep = 0
    if not hasattr(options, 'EventControl'): options.EventControl = 0
    if not hasattr(options, 'Refine'): options.Refine = 1
    if not hasattr(options, 'Verbose'): options.Verbose = 0
    if not hasattr(options, 'nargout'): options.nargout = 2  # Por defecto devuelve (T, Y, ...)

    EABS = options.AbsTol
    EREL = options.RelTol
    exactgradient = 1 if options.Gradient is not None else 0
    H0 = options.InitialStep
    eventcontrol = options.EventControl
    Refine = options.Refine
    Verbose = options.Verbose
    gradcomponents=options.GradientComponents
 
 
    IFIR = 1
    rundata = options  # Enlazar la configuración para compartirla
    tspan=np.atleast_1d(tspan)
    
 # 2. Inicialitation of the problem
    if IFIR == 1:
        IFIR = 0
        X = tspan[0]

        XEND = tspan[-1]
        ntspan = len(tspan)
        neq = len(Y)
        nout = 1
        print("\n nout", nout, X)
        
        if ntspan > 2 and options.nargout > 1:
            npoints = ntspan
            xx = np.zeros(ntspan)
            yy = np.zeros((ntspan, neq))
            xx[nout - 1] = X
            yy[nout - 1, :] = Y
        else:
            chunk = int(min(max(100, 50 * Refine), Refine + np.floor((2**13) / neq))+100)
            npoints = chunk
            xx = np.zeros(chunk)
            yy = np.zeros((chunk, neq))
            xx[nout - 1] = X
            yy[nout - 1, :] = Y
            
        tdis = []
        ydis = []
        idis = []
        stats = np.zeros(11, dtype=int)
        
        TOL = EABS + EREL * np.max(np.abs(Y))
        g0a, _, _ = switchfun(X, Y)
        g0=np.atleast_1d(g0a)
        stats[9] += 1  # stats(10) en MATLAB -> evaluaciones de la manifold function
          
        if gradcomponents.size == 0:
            gradcomponents = np.ones_like(g0)
            options.GradientComponents=gradcomponents
            
        if Verbose >= 1:
            print(f"\n Dimension: {Y.shape}")
            print(f" Manifolds: {g0.shape}")
            gra = "Yes" if exactgradient else "No"
            print(f" Exactgradient: {gra}")
            doat = "No" if options.ActionSwitch is None else "Yes"
            print(f" Doatswitch: {doat}")
            print(f" Eventcontrol: {eventcontrol}")
            print(f" Refine: {Refine}")
            print(f" Tolerances: {EABS} {EREL}")
            
        # Parámetros de precisión numérica de la máquina
        eps_1 = np.finfo(float).eps
        aux = min(eps_1**(2.0/3.0), max(2 * (EABS + EREL), 4 * eps_1))
        aux2 = np.sqrt(eps_1)
        minfortangent = [aux, aux2]
        
        integration_flow = 0
        rundata.minfortangent1 = minfortangent[0]
        rundata.minfortangent2 = minfortangent[1]
        rundata.gradcomponents = gradcomponents
        rundata.tspan = tspan
        rundata.Xend = XEND
        rundata.exactgradient = exactgradient
        rundata.nargout = options.nargout
        
        # Comprobar si el punto inicial es de discontinuidad
        inddis = 0
        if Verbose >= 1:
            print(f"\n Startind integration X= {X:22.17e}")
    
        # Llamada externa a classifypoint
        xout, yout, ff, gout, disctype, indsliding, endslid, stats = classifypoint(
            X, Y, inddis, FUN, switchfun, stats, TOL, rundata)
        if disctype == 3:
            tdis.append(X)
            ydis.append(np.copy(Y))
            idis.append(-(indsliding+1))
            stats[6] += 1  # stats(7) en MATLAB -> deslizamiento
            integration_flow = 2
        elif disctype == 1:
            idis.append(indsliding+1)
            stats[5] += 1  # stats(6)
            tdis.append(X)
            ydis=np.copy(Y)
        elif disctype == -5:
            tdis.append(X)
            ydis=np.copy(Y)
            idis.append(-(indsliding+1))
            stats[6] += 1
            integration_flow = 4
        elif disctype == -6:
            tdis.append(X)
            ydis.append(np.copy(Y))
            idis.append(-(indsliding+1))
            stats[6] += 1
            integration_flow = 6
 #       integration_flow=4
        X = xout
        Y = np.copy(yout)
        g0 = np.copy(gout)
        WRK = np.zeros((len(Y), 8))
        WRK[:, 0] = ff 
    if H0 == 0:
        norm_wrk = np.linalg.norm(WRK[:, 0])
        H0 = min((TOL / max(1e-15, norm_wrk))**(1.0 / 5.0), (XEND - X) / 10.0)
        
    H = H0
    
    # 3. Bucle Principal guiado por 'integration_flow'
    while integration_flow >= 0:
        
        # CONTROL DE FLUJO 5 o 6: Reinicio de integración
        if integration_flow in [5, 6]:
            if Verbose >= 1:
                if integration_flow == 6:
                    print(f"\n  Integration restarted, possible fail at X= {X} {H}, {disctype}")
                elif integration_flow == 5:
                    print(f"\n  First step or Integration restarted  {X}  {np.linalg.norm(Y)} {disctype}")
            xout, yout, ff, gout, disctype, indsliding, endslid, stats = classifypoint(
                X, Y, inddis, FUN, switchfun, stats, TOL, rundata
            )
            
            X = xout
            Y = np.copy(yout)
            g0 = np.copy(gout)
            WRK[:, 0] = ff
            integration_flow = 0
            if disctype == 1:
                disctype = 0
                integration_flow = 0
            elif disctype == 3:
                integration_flow = 2
                idis.append(-(indsliding+1))
            elif disctype == -5:
                integration_flow = 4
            elif disctype == -6:
                integration_flow = 6
            elif disctype == -7:
                integration_flow = 7
                
        # CONTROL DE FLUJO 0: Integración estándar (región continua)
        elif integration_flow == 0:
            if Verbose >= 1:
                no=np.linalg.norm(Y);
                print(f"\n Enter normalintegration nout, X, Y, H, {nout}, {X:22.17e}, {no:22.17e}, {H:22.17e}")     
            WRK, xx, yy, xout, yout, Hout, integration_flow, gout, xdisaprox, nout, npoints, stats = normalintegration(
                FUN, switchfun, H, X, Y, WRK, g0, xx, yy, nout, npoints, stats, rundata
            )
            X = xout
            Y = np.copy(yout)
            H = Hout
            
          
            if Verbose >= 1:
                gina, _, _ = switchfun(xout, yout)
                gin=np.atleast_1d(gina)
                disco = np.any((gin * gout) < 0)
                disco1 = gin[(gin * gout) <= 0]
                print(f"\n Exit normal integration nout, xout, yout, xdisapprox, H, disctype, disco: {nout}, {xout:22.17e} {np.linalg.norm(yout):22.17e} {xdisaprox:22.17e} {Hout:22.17e} {disctype} {int(disco)} {disco1}")   
        # CONTROL DE FLUJO 1: Localización fina de la discontinuidad hallada
        elif integration_flow == 1:
            tol = EABS + EREL * np.max(np.abs(Y))
            if Verbose >= 1:
                print(f"\n Enter FindDisc nout, X, {nout}, {X:22.17e} {xdisaprox:22.17e} {xdisaprox-X:22.17e}")
                
            xout, yout, fout, gout, Hout, integration_flow, endslid, xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats = FindDisc(
                FUN, switchfun, X, Y, H, WRK, xx, yy, xdisaprox, tdis, ydis, idis, nout, npoints, stats, tol, rundata
            )
            
            if Verbose >= 1:
                gina, _, _ = switchfun(X, Y)
                gin=np.atleast_1d(gina)
                ggouta, _, _ = switchfun(xout, yout)
                ggout=np.atleast_1d(ggouta)
                disco = np.any((gin * ggout) < 0)
                print(f"\n Exit finddisc, X, xout, yout, disctype: {X} {xout} {np.linalg.norm(yout)} {disctype} {int(disco)} {idis[-1]}")
                if disco == 0 and disctype == 1:
                    print(f"\n discontinuity not crossed:  {disctype}  {int(disco)}")
                    
            WRK[:, 0] = fout
            X = xout
            Y = np.copy(yout)
            g0 = np.copy(gout)
            H = Hout
            
            if ntspan == 2 or options.nargout == 1:
                nout += 1
                print("\n nout",nout, X)
                if nout > len(xx):
                    xx = np.append(xx, 0.0)
                    yy = np.vstack([yy, np.zeros(neq)])
                xx[nout - 1] = X
                yy[nout - 1, :] = Y
                
            if H == 0.0:
                norm_wrk = np.linalg.norm(WRK[:, 0])
                H = 0.8 * min((TOL / max(1e-15, norm_wrk))**(1.0 / 5.0), (XEND - X) / 10.0)
                
        # CONTROL DE FLUJO 2: Integración en superficie de deslizamiento (sliding)
        elif integration_flow == 2:
            if Verbose >= 1:
                print(f"\n Enter slide X, Y, H, indsliding: {X} {np.linalg.norm(Y)} {H} {indsliding}")
                
                
            WRKout, xout, yout, gout, Hout, integration_flow, xdisaprox, xx, yy, nout, npoints, stats = slide(
                FUN, switchfun, H, X, Y, WRK, g0, xx, yy, indsliding, endslid, nout, npoints, stats, rundata
            )
            
            
            if Verbose >= 1:
                print(f"\n Exit slide xout, disctype, indsliding, integration_flow: {xout} {disctype} {indsliding} {integration_flow}")
                
            X = xout
            Y = np.copy(yout)
            g0 = np.copy(gout)
            H = Hout
            WRK = np.copy(WRKout)
            
        # CONTROL DE FLUJO 3: Localización fina de discontinuidad en región de deslizamiento
        elif integration_flow == 3:
            tol = EABS + EREL * np.max(np.abs(Y))
            if Verbose >= 1:
                print(f"\n Enter finddiscpro  X, disctype, indsliding: {X} {disctype} {indsliding}")
                
            WRKout, xout, yout, gout, Hout, integration_flow, xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats = FindDiscpro(
                FUN, switchfun, H, X, Y, WRK, xx, yy, xdisaprox, tdis, ydis, idis, indsliding, nout, npoints, stats, tol, rundata
            )
            
            if Verbose >= 1:
                print(f"\n Exit finddiscpro  xout, disctype, indsliding: {xout} {disctype} {indsliding}")
                
            WRK = np.copy(WRKout)
            X = xout
            Y = np.copy(yout)
            g0 = np.copy(gout)
            
            if ntspan == 2 or options.nargout == 1:
                nout += 1
                print("\n nout",nout, X)
                if nout > len(xx):
                    xx = np.append(xx, 0.0)
                    yy = np.vstack([yy, np.zeros(neq)])
                xx[nout - 1] = X
                yy[nout - 1, :] = Y
                
            H = Hout / 4.0
            if Hout == 0.0:
                norm_wrk = np.linalg.norm(WRK[:, 0])
                H = 0.8 * min((TOL / max(1e-15, norm_wrk))**(1.0 / 5.0), (XEND - X) / 10.0)
                
        # CONTROL DE FLUJO 4 o 7: Codimensión > 2 o superficies tangenciales. Se recurre a solver estándar
        elif integration_flow in [4, 7]:
            if Verbose >= 0:
                if integration_flow == 4:
                    ndis = len(indsliding) if isinstance(indsliding, (list, np.ndarray)) else 1
                    print(f"\n co-dimension {ndis} Filippov point at X= {X}, manifolds= {indsliding}")
                elif integration_flow == 7:
                    print(f"\n Tangent surfaces at X= {X}, manifolds= {indsliding}")
                    
            EABS = max(1e-5, EABS)
            EREL = max(1e-5, EREL)
            print("\n  The integration does not control the discontinuities and can be very slow from this point")
            
            paso = (XEND - tspan[0]) / 1.0
            xendnext = min(X + paso, XEND)
            
            # Reemplazo de ode45 de MATLAB con solve_ivp de scipy
            if ntspan > 2 and options.nargout > 1:
                tspan_mask = tspan > X
                tspanlast = tspan[tspan_mask]
                tspanlast = tspanlast[tspanlast <= xendnext + 1e-10]
                tspanlast = np.insert(tspanlast, 0, X)
                nadd = len(tspanlast)
                
                # Integrar con solve_ivp usando el método RK45 (equivalente a ode45)
                ffout, xxx, yyy, xout, yout, Hout, nout, npoints, stats= rkintegration(
                    FUN, switchfun,H, X, Y, EABS, EREL, xx, yy, nout, npoints, XEND, WRK, stats, Verbose)

                if nadd == 2:
                    # Rellenar matrices dinámicas xx e yy
                    while nout + 1 > len(xx):
                        xx = np.append(xx, 0.0)
                        yy = np.vstack([yy, np.zeros(neq)])
                    xx[nout] = xxx[-1]
                    yy[nout, :] = yyy[-1, :]
                else:
                    while nout + nadd - 1 > len(xx):
                        xx = np.append(xx, np.zeros(chunk))
                        yy = np.vstack([yy, np.zeros((chunk, neq))])
                    xx[nout : nout + nadd - 1] = xxx[1:]
                    yy[nout : nout + nadd - 1, :] = yyy[1:, :]
                    
                nout = nout + nadd - 1
                X = xx[nout - 1]
                Y = np.copy(yy[nout - 1, :])
                print("\n nout",nout, X)
            else:
                tspanlast = [X, xendnext]
                ffout, xx, yy, xout, yout, Hout, nout, npoints,stats= rkintegration(
                    FUN, switchfun,H, X, Y, EABS, EREL, xx, yy, nout, npoints, XEND, WRK, stats, Verbose)

                # Concatenar resultados
                X = xx[nout-1]
                Y = np.copy(yy[nout-1, :])
      
            if abs(X - XEND) < 1e-14:
                integration_flow = -5
                
            else:
                integration_flow = 5
                
    # 4. Finalización y formato de salida
    if Verbose >= 1:
        if integration_flow == -3:
            print("\n Integration ended at a terminal switching point \n")
        elif integration_flow == -4:
            print("\n Integration ended at a co-dimension 3 switching point \n")
        elif integration_flow == -1:
            print("\n Integration ended, minimum step size reached \n")
            
    # Recortar los vectores al tamaño real calculado
    xx = xx[:nout]
    yy = yy[:nout, :]
    
    print(f"\n nout= {nout}, ndis= {len(idis)}")
    
    # Conversiones a vectores tipo columna de MATLAB en caso necesario (.T)
    tdis = np.array(tdis)
    ydis = np.array(ydis)
    idis = np.array(idis)
    
    
    if options.nargout == 1:
        sol = Struct()
        sol.x = xx
        sol.y = yy.T
        sol.xd = tdis
        sol.yd = ydis
        sol.id = idis
        sol.stats = stats
        return sol
    else:
        return xx, yy, tdis, ydis, idis, stats

def normalintegration(FUN, switchfun, H, X, Y, WRK, gxy, xx, yy, nout, npoints, stats, rundata):
    """
    Traducido de MATLAB a Python.
    Nota: Se asume que 'xx', 'yy' y 'stats' son arreglos de NumPy (np.array).
    'rundata' es un objeto o clase que contiene los atributos de configuración.
    """
    
    # Extraer datos de rundata (asumiendo formato de objeto/clase con atributos)
    eventcontrol = rundata.EventControl
    Verbose = rundata.Verbose
    Refine = rundata.Refine
    XEND = rundata.Xend
    tspan = rundata.tspan
    EABS = rundata.AbsTol
    EREL = rundata.RelTol
    solyes = rundata.nargout
    
    advance = True
    REJECT = False
    
    xdis = XEND
    xout = X
    
    H = min(H, XEND - X)
    Hout = H
    checkall = eventcontrol
    discdetected = 0
    irestart = 0
    
    stats[9] += 1  # MATLAB stats(10) -> Python stats[9]
    gxyout = np.copy(gxy)
    
    # Cálculo de 'paso' resguardando división por cero en norm
    norm_WRK_0 = np.linalg.norm(WRK[:, 0])
    max_denom = max(0.001, norm_WRK_0)
    paso = 100000 * max(1.0 * np.finfo(float).eps * X, max(np.finfo(float).eps * Y) / max_denom)
    
    if Verbose >= 3:
        f0 = WRK[:, 0]
        yaux = Y + paso * f0
        faux = FUN(X + paso, yaux)
        stats[8] += 1  # stats(9) -> stats[8]
        gnorm = np.linalg.norm(faux - f0)
        gauxa, _, _ = switchfun(X + paso, yaux)
        gaux=np.atleast_1d(gauxa)
        stats[9] += 1  # stats(10) -> stats[9]
        gsign = np.any((gaux * gxy) < 0)
        
        if gsign or gnorm > 1:
            print(f"\n gsign, gnorm: {int(gsign)} {X} {gnorm} {paso}")
            print(f"\n gxy:  {' '.join(f'{val:28.25g}' for val in gxy)}")
            print(f"\n gyx:  {' '.join(f'{val:28.25g}' for val in gaux)}")
            print('\n bad crossed or too close discontinuities')
            xout = X

    neq = Y.shape[0]
    chunk = int(min(max(100, 50 * Refine), Refine + np.floor((2**13) / neq))+100)
    
    while advance:
        if H < max(5 * np.finfo(float).eps * X, np.finfo(float).eps * 0.001):
            if Verbose >= 0:
                print(f"\n Minimum step size  h={H} attained at X= {X} \n  Integration stopped \n")
            yout = np.copy(Y)
            WRKout = np.copy(WRK)
            integration_flow = -1
            return WRKout, xx, yy, xout, yout, Hout, integration_flow, gxyout, xdis, nout, npoints, stats
        
        # Llamada a la función externa RKNEW
    
   
        XPH, Y1, WRKout, ERR, disctype, gxyout, xdis, stagedis, stats = RKNEW(
            FUN, switchfun, X, H, Y, XEND, WRK, checkall, gxy, stats, EABS, EREL
        )
        
        if Verbose >= 4:
            print(f"\n primera etapa {WRKout[:, 0]}")
            print(f"\n segunda etapa {WRKout[:, 1]}")
            
        TOL = EABS + EREL * np.max(np.abs(Y1))
        
        if disctype == 0:  # No discontinuity detected
            if ERR <= TOL:
                # Paso aceptado
                if Verbose >= 2:
                    print(f"\n Paso aceptado H, X, XPH {H:20.15e}   [{X:20.15e}, {XPH:20.15e}]")
                
                stats[0] += 1  # stats(1) -> stats[0]
                WRK = np.copy(WRKout)
                
                # tspan tiene más de 2 elementos y solyes > 1
                if tspan.ndim > 0 and tspan.shape[0] > 2 and solyes > 1:
                    tnew = np.where((tspan > X) & (tspan <= XPH))[0]
                    for ii in range(len(tnew)):
                        nout += 1
                        xtspan = tspan[tnew[ii]]
                        Y2 = ESTIRANEW(X, Y, WRK, H, xtspan)
                        xx[tnew[ii]] = xtspan
                        yy[tnew[ii], :] = Y2
                        print("\n nout tspan",nout, xtspan)
                else:
                    if nout + Refine+1 > npoints:
                        npoints += chunk
                        xx = np.append(xx, np.zeros(chunk))
                        yy = np.vstack([yy, np.zeros((chunk, neq))])
                        
                    if Refine >= 2:
                        for irefine in range(1, Refine):
                            xrefine = X + (XPH - X) * irefine / Refine
                            Y2 = ESTIRANEW(X, Y, WRK, H, xrefine)
                            nout += 1
                            xx[nout - 1] = xrefine
                            yy[nout - 1, :] = Y2
                            print("\n nout Refine",nout, xrefine)
                            
                    nout += 1
                    xx[nout - 1] = XPH
                    yy[nout - 1, :] = Y1
                    print("\n nout normal",nout, XPH)
                
                Y = np.copy(Y1)
                X = XPH
                gxy = np.copy(gxyout)
                
                # Actualizar el tamaño del paso
                FAC = min(0.9 * (TOL / (ERR + 1e-17))**(1.0 / 5.0), 2.0)
                if REJECT:
                    FAC = min(FAC, 1.0)
                    
                H = FAC * H
                REJECT = False
                checkall = eventcontrol
                
                if discdetected == 1:
                    checkall = eventcontrol + 1
                    discdetected = 0
                    
                if (X - XEND) + 5.0 * max(np.finfo(float).eps * 1.0, np.finfo(float).eps * X) > 0.0:
                    X = XEND
                    yout = np.copy(Y)
                    xout = X
                    integration_flow = -5
                    return WRKout, xx, yy, xout, yout, Hout, integration_flow, gxyout, xdis, nout, npoints, stats
                
                WRK[:, 0] = FUN(XPH, Y1)
                stats[8] += 1  # stats(9) -> stats[8]
                
                if (X + H - XEND) > 0.0:
                    H = XEND - X
                irestart = 0
                
            else:
                # Paso rechazado
                if Verbose >= 2:
                    print(f"\nPASO rechazado {H}   [{X:.20e}, {X+H}]")
                FAC = max(0.9 * (TOL / (ERR + 1e-12))**(1.0 / 5.0), 0.10)
                REJECT = True
                H = FAC * H
                stats[1] += 1  # stats(2) -> stats[1]
                
        else:  # Discontinuity detected at this step.
            if Verbose >= 2:
                print(f"\n Possible discontinuity {H:20.15e}   [{X:20.15}, {X+H:20.15e}] {disctype} {stagedis}")
            irestart += 1
            stats[4] += 1  # stats(5) -> stats[4]
            checkall = eventcontrol + 1
            
            if stagedis > 7 and ERR <= TOL:
                xout = X
                yout = np.copy(Y)
                Hout = H
                integration_flow = 1
                return WRKout, xx, yy, xout, yout, Hout, integration_flow, gxyout, xdis, nout, npoints, stats
            elif stagedis > 7 and ERR > TOL:
                H = 0.5 * H
                REJECT = True
            elif irestart >= 6 or H < max(100 * np.finfo(float).eps * X, np.finfo(float).eps * 0.01):
                integration_flow = 5
                xout = X + paso
                yout = Y + paso * WRK[:, 0]
                return WRKout, xx, yy, xout, yout, Hout, integration_flow, gxyout, xdis, nout, npoints, stats
            else:
                if irestart >= 4 and X == xout and stagedis <= 2 and REJECT:
                    if Verbose >= 0:
                        print(f"\n  Warning!!, bad restarting?? {H}  {X:.20e} {ERR}  {stagedis}")
                    integration_flow = 6
                    xout = X
                    yout = np.copy(Y)
                    npasos = 2
                    input("Presiona Enter para continuar (Simulación de pause de MATLAB)...")
                    
                    for ii in range(1, npasos + 1):
                        xout = xout + (4**ii) * paso
                        yout = yout + (4**ii) * paso * WRK[:, 0]
                        WRK[:, 1] = FUN(xout, yout)
                        stats[8] += 1
                    return WRKout, xx, yy, xout, yout, Hout, integration_flow, gxyout, xdis, nout, npoints, stats
                
                H = min(xdis - X, H)  # Ajuste de seguridad frente al original min(xdis-X)
                REJECT = True
                
    return WRKout, xx, yy, xout, yout, Hout, integration_flow, gxyout, xdis, nout, npoints, stats

def FindDisc(FUN, switchfun, X, Y, H, WRK, xx, yy, xdisaprox, tdis, ydis, idis,
              nout, npoints, stats, tol, rundata):
    """
    Traducción de FindDisc de MATLAB a Python.
    Se asume que 'rundata' puede ser un objeto o una clase con atributos,
    y que funciones externas como ESTIRANEW y classifypoint están definidas.
    """
    # -----------------------------------------------------------------------
    # Variables de configuración desde rundata
    # -----------------------------------------------------------------------
    Verbose = rundata.Verbose
    xend = rundata.Xend
    tspan = np.atleast_1d(rundata.tspan)  # Asegurar que sea un array
    Refine = rundata.Refine
    doatswitch = rundata.ActionSwitch
    solyes = rundata.nargout  # Simulación de nargout si es necesario
    indsliding = -1
    endslid = 1.0
    Hout = max(H, 1.e-9)
    ydis=np.asarray(ydis)
    neq = Y.shape[0]

    fout = WRK[:, 0].copy() if WRK.ndim > 1 else WRK.copy() # Primera columna de WRK
    # -----------------------------------------------------------------------
    # Lower extreme of the interval
    # -----------------------------------------------------------------------
    if xdisaprox > X + H:
        X0 = X + H
        X1 = min(xdisaprox, xend)
    else:
        X0 = X + 0.9 * H
        X1 = min(xdisaprox, xend)

    ynew0 = ESTIRANEW(X, Y, WRK, H, X0)
    gg0a, _, _ = switchfun(X0, ynew0)
    gg0=np.atleast_1d(gg0a)
    stats[9] += 1  # MATLAB stats(10) -> Python stats[9]
    
    value0 = np.min(np.abs(gg0[gg0 != 0]))
    valuea = value0

    if np.any(np.abs(gg0) == 0):
        eps_X = np.spacing(X)
        eps_1 = np.spacing(1.0)
        step_min = min(1.e-10, max([tol / 100.0, eps_X, 100.0 * eps_1]))
        
        xout = X + step_min
        yout = Y + step_min * WRK[:, 0]
        fout = FUN(xout, yout)
        stats[8] += 1  # MATLAB stats(9) -> Python stats[8]
        integration_flow = 0
        gout, _, _ = switchfun(xout, yout)
        endslid = 1.0
        
        if Verbose >= 1:
            print(f"\n Possible repeated discon !!!! {X} {X0}")
            
        return (xout, yout, fout, gout, Hout, integration_flow, endslid, 
                xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)

    # -----------------------------------------------------------------------
    # Upper extreme of the interval
    # -----------------------------------------------------------------------
    ynew = ESTIRANEW(X, Y, WRK, H, X1)
    xnew = X1
    gg1in, isterminal, direction = switchfun(X1, ynew)
    gg1=np.atleast_1d(gg1in)
    direction=np.atleast_1d(direction)
    stats[9] += 1  # stats(10)
    
    # Lógica de vectores elemento a elemento
    gcondpa = (gg1 * gg0 <= 0) * (gg1 * direction >= 0)
    gcondp=np.atleast_1d(gcondpa)

    if np.any(gcondp):  # Checking if there is a discontinuity in the interval
        value1 = -np.max(np.abs(gg1[gcondp == True]))
        ind = np.atleast_1d(np.where(gcondp == True)).flatten()
        if Verbose >= 1:
            print(f"\n discon index: {ind}")
    else:
        eps_1 = np.spacing(1.0)
        if np.min(np.abs(gg1)) < 10.0 * eps_1 and np.abs(X1 - xend) <= 10.0 * eps_1:
            inddis = np.where(np.abs(gg1) < 10.0 * eps_1)[0]
            idis.append(inddis)
            stats[5] += 1  # stats(6)
            if Verbose >= 2:
                print(f"\n Discontinuity at X1: {X1} {ynew[0]}")
            integration_flow = -5
            xout = xend
            yout = ynew
            endslid = 1.0
            gout = gg1
            return (xout, yout, fout, gout, Hout, integration_flow, endslid, 
                    xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)
                    
        elif np.min(np.abs(gg1)) < 10.0 * eps_1:
            inddis = np.where(np.abs(gg1) < 10.0 * eps_1)[0]
            idis.append(inddis)
            stats[5] += 1  # stats(6)
            if Verbose >= 2:
                print(f"\n Discontinuity at X1: {X1} {ynew[0]}")
            integration_flow = 5
            xout = X1
            yout = ynew
            gout = gg1
            endslid = 1.0
            return (xout, yout, fout, gout, Hout, integration_flow, endslid, 
                    xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)
        else:
            if Verbose >= 0:
                print(f"\n Warning failed discon !!!! {X} {gg0}")
                print(f"\n Warning failed discon !!!! {H} {gg1}")
            xout = X1
            yout = ynew
            gout = gg1
            integration_flow = 0
            Hout = H / 2.0
            endslid = 1.0
            fout = FUN(X1, ynew)
            stats[8] += 1  # stats(9)
            return (xout, yout, fout, gout, Hout, integration_flow, endslid, 
                    xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)

    # -----------------------------------------------------------------------
    # Verificaciones de paso mínimo
    # -----------------------------------------------------------------------
    eps_X0 = np.spacing(X0)
    eps_1 = np.spacing(1.0)
    if np.abs(X1 - X0) / 2.0 < 1.0 * max(eps_X0, eps_1):
        xnew = X0
        if Verbose >= 0:
            print(f"\n Warning very small step size in finddisc !!!! {X} {X0} {X1} {H} {X1-X0}")
    elif np.abs(valuea) < 1.0 * eps_1:
        xnew = X0
        if Verbose >= 2:
            print(f"\n Discontinuity at X0 !!!! {X} {X0} {X1} {Y[0]} {ynew[0]} {X1-X0}")

    # -----------------------------------------------------------------------
    # Secant-Bisection Method's loop
    # -----------------------------------------------------------------------
    xx1 = X0
    xx2 = X1
    ii = 0 # En Python empezamos en 0, iteramos hasta < 60 (equivalente a 1 a 60 en MATLAB)
    while np.abs(xx2 - xx1) / 2.0 >= 1.0 * max(np.spacing(xx1), eps_1) and valuea >= 1.0 * eps_1 and ii < 60:
        if Verbose >= 2 and ii == 0:
            print('\n Refining discontinuity with secant method')          
        if abs(value1-value0)<1.e-16:
            xnew=X1-1.e-16
        else:
            xnew = X1 - value1 * (X1 - X0) / (value1 - value0)       
            
        if xnew >= xx2 or xnew <= xx1 or np.abs(value1 - value0) < 1.e-14 or ii > 20:
            xnew = (xx1 + xx2) / 2.0  # Bisección si la secante falla
            
        ynew = ESTIRANEW(X, Y, WRK, H, xnew)
        gnewa, isterminal, direction = switchfun(xnew, ynew)
        gnew=np.atleast_1d(gnewa)
        isterminal=np.atleast_1d(isterminal)
        direction=np.atleast_1d(direction)
        if np.isscalar(isterminal):
            isterminal=np.array([isterminal])
        if np.isscalar(direction):
            direction=np.array([direction])
        stats[9] += 1  # stats(10)
        gcondp = (gnew * gg0 <= 0) * (gnew * direction >= 0)

        if np.any(gcondp):
            value = -np.max(np.abs(gnew[gcondp == True]))
            ind = np.atleast_1d(np.where(gcondp == True))
            ind=ind.flatten()
            xx2 = xnew
        else:
            value = np.min(np.abs(gnew[ind]))
            valuea = value
            xx1 = xnew

        # Actualizar históricos para la secante
        X0 = X1
        value0 = value1
        X1 = xnew
        value1 = value
        ii += 1

    if ii >= 60:
        print(f'\n Secant method attained the maximum of iterations {ii}')
    if Verbose >= 1:
        print(f'\n  discontinuity found at X={xnew:22.17}, {np.linalg.norm(ynew)}')
    tdis = np.append(tdis, xnew)
    ydis = np.vstack([ydis, ynew]) if ydis.size else np.array([ynew])
    # -----------------------------------------------------------------------
    # Guardar resultados en los históricos xx e yy
    # -----------------------------------------------------------------------
    if tspan.shape[0] > 2 and solyes > 1:
        tnew = np.where((tspan > X) & (tspan <= xnew))[0]
        for i in range(tnew.shape[0]):
            nout += 1
            xtspan = tspan[tnew[i]]
            Y2 = ESTIRANEW(X, Y, WRK, H, xtspan)
            xx[tnew[i]] = xtspan
            yy[tnew[i], :] = Y2
            print("\n nout",nout, xtspan)
    else:
        if Refine >= 2:
            for irefine in range(1, Refine):
                xrefine = X + (xnew - X) * irefine / Refine
                Y2 = ESTIRANEW(X, Y, WRK, H, xrefine)
                nout += 1
                xx[nout - 1] = xrefine
                yy[nout - 1, :] = Y2
                print("\n nout",nout, xrefine)
        nout += 1
        if nout > len(xx):
           xx = np.append(xx, 0.0)
           yy = np.vstack([yy, np.zeros(neq)])
        xx[nout - 1] = xnew
        yy[nout - 1, :] = ynew
        print("\n nout",nout, xnew)


    inddis = ind[0]
    

    # -----------------------------------------------------------------------
    # Clasificación del punto numérico hallado
    # -----------------------------------------------------------------------
 
    if np.any(isterminal[ind] < 0):
        yout = doatswitch(xnew, ynew)
        xout = xnew
        gout = gg0
  #      inddis = 0
        idis.append(inddis+1)
        stats[5] += 1  # stats(6)
        integration_flow = 5
        if Verbose >= 1:
            print(f'\n Discontinuity restarting point at:  {xnew}  {np.linalg.norm(yout)} {ind}')
        return (xout, yout, fout, gout, Hout, integration_flow, endslid, 
                xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)
    else:
        # Validación de índice único
        if (ind.ndim > 0 and ind.shape[0] != 1) and Verbose >= 0:
            print(f'\n Warning, multiple dicontinuity index= !!! {ind}')
            
        if np.any((isterminal[ind] == 1) & ((direction[ind] * gg0[ind] <= 0) == 1)):
            integration_flow = -2
            idis.append(inddis+1)
            stats[5] += 1  # stats(6)
            if Verbose >= 1:
                print('\n the switching point ends the integration')
            xout = xnew
            yout = ynew
            gout = gnew
            Hout = H
            return (xout, yout, fout, gout, Hout, integration_flow, endslid, 
                    xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)

        # Llamar a la función externa classifypoint
        xout, yout, fout, gout, disctype, indsliding, endslid, stats = classifypoint(
            xnew, ynew, inddis, FUN, switchfun, stats, tol, rundata
        )

    # Convertir a array de 1D si disctype devuelve un escalar para evitar errores de indexación
    disctype = np.atleast_1d(disctype)
    endslid_arr = np.atleast_1d(endslid)
    if disctype[0] == 3 and np.abs(endslid_arr[0]) <= 1.e-5:
        idis.append(-(inddis+1))
        stats[6] += 1  # stats(7)
        if Verbose >= 1:
            print(f'\n Tangent point at:  {xnew}  {ynew[0]}')
        integration_flow = 2
    elif disctype[0] == 3 and np.any(endslid_arr < 0):
        idis.append(-(inddis+1))
        stats[6] += 1  # stats(7)
        if Verbose >= 2:
            print(f'\n Filippov point {xnew} {ynew[0]}')
        integration_flow = 2
    elif disctype[0] == 1:
        idis.append(inddis+1)
        stats[5] += 1  # stats(6)
        if Verbose >= 1:
            print(f'\n Transversal discontinuity, exit at: {xnew} {ynew[0]}')
        integration_flow = 0
    elif disctype[0] == -5:
        integration_flow = 4
    elif disctype[0] == 0:
        idis.append(inddis+1)
        stats[5] += 1  # stats(6)
        integration_flow = 0
    else:
        if Verbose >= 0:
            print(f'\n Other discontinuity, exit at {xnew} {ynew[0]}, {endslid} {disctype} {inddis}')
        idis.append(inddis+1)
        stats[5] += 1  # stats(6)
        integration_flow = 0

    Hout = H
    if tspan.shape[0] > 2 and solyes > 1:
        tnew = np.where((tspan > xnew) & (tspan <= xout))[0]
        for i in range(tnew.shape[0]):
            nout += 1
            xtspan = tspan[tnew[i]]
            Y2 = (ynew + yout) / 2.0
            xx[tnew[i]] = xtspan
            yy[tnew[i], :] = Y2
            print("\n nout",nout, xtspan)
    return (xout, yout, fout, gout, Hout, integration_flow, endslid, 
            xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)
#
#   End of Findisc

def RKNEW(FUN, switchfun, X, H, Y, XEND, WRK, checkall, gxy, stats, EABS, EREL):
    # Asegurar que las variables estructuradas sean arreglos de NumPy
    Y = np.asarray(Y, dtype=float)
    WRK = np.asarray(WRK, dtype=float)
    gxy = np.asarray(gxy, dtype=float)
    stats = np.asarray(stats, dtype=int)
    
    WRKout = WRK.copy()
    
    # Matriz A original adaptada a dimensiones Python (7x7)
    A = np.array([
        [0.108029, 0, 0, 0, 0, 0, 0],
        [0.0405108750, 0.1215326250, 0, 0, 0, 0, 0],
        [0.0607663125, 0.0, 0.1822989375, 0, 0, 0, 0],
        [0.297105016920909930065219608547, 0.0, -0.969346938437952801300701607363, 1.18312292151704287123548199882, 0, 0, 0],
        [-0.445767719936558689939662812961, 0, 2.74214758450584659845562912716, -2.34407023022786831138086780967, 0.801260365658580402864901495470, 0, 0],
        [0.166870931891601717285119862416, 0, -0.735371113362101801736678191285, 1.20597299548071395090925872364, -0.136798000000000000000000000000, 0.399325185989786133542299605231, 0]
    ])
    
    C = np.array([0.0, 0.108029, 324087.0/2000000.0, 972261.0/4000000.0, 0.510881, 0.75357, 0.9])
    
    B = np.array([
        0.0835287062817292866236614389718, 0.0, 0.0, 0.306545998706544449984758902241,
        0.267965179973423262607667345489, 0.130936983245107245109927623181, 0.211023131793195755673984690118
    ])
    
    B1 = np.array([
        -0.000882201938392711047700974918454, 0, 0, 0.632177414748733110758337278349,
        -0.282232701759180326509200083357, 0.650937488948839926798563779927, 0
    ])
    
    disctype = 0
    xdis = X + H
    XPH = X
    gxyout = np.atleast_1d(gxy)
    gxya = np.atleast_1d(gxy)
    stagedis = 8
    
    # MATLAB 2:7 equivale a range(1, 7) en Python
    for K in range(1, 7):
        HH = H * C[K]
        Y1 = Y + (H * (WRK[:, :K] @ A[K-1, :K]))
        
        if checkall >= 1:
            gyxin, _, direction = switchfun(X + HH, Y1)
            gyx=np.atleast_1d(gyxin)
            stats[9] += 1  # stats(10) -> indice 9
            auxa = (np.sign(gyx) * np.sign(gxy)) <= 0
            auxb = (gyx * direction) >= 0
            if np.any(auxa * auxb):
                disctype = 1
                xdis = X + H * C[K]
                stagedis = K +1 # Para mantener la semántica de MATLAB de etapas 1-8
                ERR = 2.0
                
                # Evitar división por cero si gxya == gyx
                denom = gxya[auxa] - gyx[auxa]
                xxdis = np.min(gxya[auxa] / np.where(denom == 0, 1e-15, denom))
                xxdis = np.min(gxya[auxa] / (gxya[auxa] - gyx[auxa]))
                xxdis = max(
                    X + H * C[K-1] + H * (C[K] - C[K-1]) / 10,
                    min(X + H * C[K-1] / 0.92 + H * (C[K] - C[K-1]) * xxdis / 0.92, xdis)
                )
                xdis = xxdis
                gxyout = gyx.copy()
                return XPH, Y1, WRKout, ERR, disctype, gxyout, xdis, stagedis, stats
            
            gxya = gyx.copy()
            
        WRK[:, K] = FUN(X + HH, Y1)
        stats[8] += 1  # stats(9) -> indice 8
    EST = 0.0
    Y1 = Y.copy()
    for K in range(7):
        EST = EST + H * WRK[:, K] * (B[K] - B1[K])
        Y1 = Y1 + H * WRK[:, K] * B[K]
        
    ERR = float(np.max(np.abs(EST)))
    TOL = EABS + EREL * np.max(np.abs(Y1))
    if ERR > TOL:
        XPH = X + H
        return XPH, Y1, WRKout, ERR, disctype, gxyout, xdis, stagedis, stats
        
    ynew = ESTIRANEW(X, Y, WRK, H, X + C[6] * H)
    gg1a, _, direction = switchfun(X + C[6] * H, ynew)
    gg1=np.atleast_1d(gg1a)
    stats[9] += 1
    
    auxa = (np.sign(gg1) * np.sign(gxy)) <= 0
    auxb = (gg1 * direction) >= 0
    if np.any(auxa * auxb):
        xdis = X + C[6] * H
        disctype = 1
        stagedis = 7
        gxyout = gg1
        ERR = 2.0
        return XPH, Y1, WRKout, ERR, disctype, gxyout, xdis, stagedis, stats
        
    g1a, _, direction = switchfun(X + H, Y1)
    g1=np.atleast_1d(g1a)
    stats[9] += 1
    WRKout = WRK.copy()
    ERR = float(np.max(np.abs(EST)))
    stats[9] += 1
    auxa = (np.sign(g1) * np.sign(gxy)) <= 0
    auxb = (g1 * direction) >= 0
    if np.any(auxa * auxb):
        xdis = X + H
        disctype = 1
        stagedis = 8
        gxyout = g1
        return XPH, Y1, WRKout, ERR, disctype, gxyout, xdis, stagedis, stats
        
    if ERR > TOL / 10 or (X + 1.1 * H) > XEND:
        XPH = X + H
        gxyout = g1
        return XPH, Y1, WRKout, ERR, disctype, gxyout, xdis, stagedis, stats
        
    xnew = min(X + 1.1 * H, XEND)
    ynew = ESTIRANEW(X, Y, WRKout, H, xnew)
    gyx, _, direction = switchfun(xnew, ynew)
    gyx=np.atleast_1d(gyx)
    direction=np.atleast_1d(direction)
    stats[9] += 1
    gnew = gyx.copy()
    auxa = (np.sign(gyx) * np.sign(gxy)) <= 0
    auxb = (gyx * direction) >= 0
    auxend = np.any(auxa * auxb)
    
    if checkall == 0:
        if ERR > TOL or auxend:
            
            for K in range(1, 7):
                Y2 = Y + (H * (WRK[:, :K] @ A[K-1, :K]))
                gyxa, _, direction = switchfun(X + H * C[K], Y2)
                gyx=np.atleast_1d(gyxa)
                stats[9] += 1
                auxa = (np.sign(gyx) * np.sign(gxy)) <= 0
                auxb = (gyx * direction) >= 0
                
                if np.any(auxa * auxb):
                    disctype = 1
                    stagedis = K + 1
                    denom = gxya[auxa] - gyx[auxa]
                    xxdis = np.min(gxya[auxa] / np.where(denom == 0, 1e-15, denom))
                    xxdis = min(X + H * C[K-1] / 0.92 + H * (C[K] - C[K-1]) * xxdis / 0.92, X + H * C[K])
                    xdis = xxdis
                    gxyout = gyx
                    return XPH, Y1, WRKout, ERR, disctype, gxyout, xdis, stagedis, stats
                gxya = gyx.copy()
                
    if auxend:
        xdis = xnew
        gxyout = gyx
        disctype = 1
        stagedis = 8
        return XPH, Y1, WRKout, ERR, disctype, gxyout, xdis, stagedis, stats
        
    if checkall >= 2 and ERR <= TOL:
        for icheck in range(1, 6 * checkall + 1):
            xcheck = X + H * icheck / (6 * checkall + 1) + 0.1 * H
            Y2 = ESTIRANEW(X, Y, WRKout, H, xcheck)
            gyxa, _, direction = switchfun(xcheck, Y2)
            gyx=np.atleast_1d(gyxa)
            stats[9] += 1
            auxa = (np.sign(gyx) * np.sign(gxy)) <= 0
            auxb = (gyx * direction) >= 0
            if np.any(auxa * auxb):
                disctype = 1
                xdis = xcheck
                gxyout = gyx
                return XPH, Y1, WRKout, ERR, disctype, gxyout, xdis, stagedis, stats
                
    if ERR <= TOL:
        gxyout = gnew
        
    XPH = X + H
    WRKout = WRK.copy()
    
    return XPH, Y1, WRKout, ERR, disctype, gxyout, xdis, stagedis, stats

#
#  End of RKNEW
#
def ESTIRANEW(X, Y, WRK, H, XA):
    """
    This function uses the continuous extension of the step [X, X+H] 
    to get an approximation of order five at the point XA, 
    with XA in [X, X+H+0.3*H].
    """
    # Asegurar que los datos de entrada sean arreglos de NumPy
    Y = np.asarray(Y, dtype=float)
    WRK = np.asarray(WRK, dtype=float)
    
    ta = (XA - X) / H
    t=ta
    
    # Inicializar el vector de coeficientes co (7 elementos, índices 0 a 6)
    co = np.zeros(7, dtype=float)
    
    
    co[0] = t * (1.0 \
            + t * (-4.25482623019970324911717856546 \
            + t * (8.11020341573537879642017144065 \
            + t * (-7.14661687329236236287641040756 \
            + t * 2.37476839403841610219707897135))))
            
    co[1] = 0.0
    co[2] = 0.0
    
    co[3] = t**2 * (7.93551021432329404292127641510 \
            + t * (-23.2538532563247245209857499444 \
            + t * (24.7860177529639518319024202303 \
            + t * (-9.16112871225597690385318779877))))
            
    co[4] = t**2 * (-6.37926427286141426113267663301 \
            + t * (27.8656872767412852432264442101 \
            + t * (-36.6974115226181735878538005999 \
            + t * 15.4789536987117258683677003683)))
            
    co[5] = t**2 * (4.08744193814364535150016196301 \
            + t * (-19.5724043305317507841599511570 \
            + t * (30.2452973037891945586149965449 \
            + t * (-14.6293979281559818808452797278))))
            
    co[6] = t**2 * (-1.38886164940582188417158317964 \
            + t * (6.85036689437981126549908545057 \
            + t * (-11.1872866608426104397872057677 \
            + t * 5.93680454766181681413368818688)))
            
    # Multiplicación matricial: WRK[:, :7] toma las primeras 7 columnas (0 a 6)
    # El operador @ multiplica la matriz por el vector de coeficientes co
    cont = WRK[:, :7] @ co
    
    # Cálculo de la aproximación final en XA
    Y1 = Y + H * cont
    
    return Y1
#
#  End of ESTIRANEW
#

def graddif(switchfun, t, y, ind, tol, stats, v=None):
    """
    Traducción de la función graddif de MATLAB a Python utilizando NumPy.
    Calcula el gradiente con respecto a las variables espaciales (y).
    """
    eps = sys.float_info.epsilon
    
    # Emulación de nargin
    if v is not None:
        N = 1
        vv = np.array(v) # Aseguramos que sea un array de NumPy
        # Si v es un vector unidimensional, lo convertimos en columna de 2D para mantener consistencia
        if vv.ndim == 1:
            vv = vv.reshape(-1, 1)
    else:
        # size(y, 1) en MATLAB obtiene el número de filas de y.
        # En Python asumimos que 'y' es un array de NumPy (o una lista).
        y_arr = np.asarray(y)
        N = y_arr.shape[0]
        vv = np.eye(N)
    
    gxd = np.zeros(N)
    
    for it in range(N):
 
        # En Python las columnas se extraen con [:, it]
        v_col = vv[:, it]
        
        if tol > 1.e-5:
            e = np.sqrt(eps)
            gy = switchfun(t, y)[0]
            stats[9] += 1
            z = y + e * v_col
            gz = switchfun(t, z)[0]
            stats[9] += 1
            gxdit = (gz[ind] - gy[ind]) / e
            
        elif tol > 1.e-8:
            e = eps**(1/3)
            gz1 = switchfun(t, y + e * v_col / 2)[0]
            gz2 = switchfun(t, y - e * v_col / 2)[0]
            stats[9] += 2
            gxdit = (gz1[ind] - gz2[ind]) / e
  
            
        elif tol > 1.e-11:
            e = 0.9 * (eps**(1/6))
            gz1 = switchfun(t, y + e * v_col)[0]
            gz1=np.atleast_1d(gz1)
            gz2 = switchfun(t, y + e * v_col / 2)[0]
            gz2=np.atleast_1d(gz2)
            gz3 = switchfun(t, y - e * v_col / 2)[0]
            gz3=np.atleast_1d(gz3)
            gz4 = switchfun(t, y - e * v_col)[0]
            gz4=np.atleast_1d(gz4)
            stats[9] += 4
            gxdit = (-gz1[ind] + 8 * gz2[ind] - 8 * gz3[ind] + gz4[ind]) / (6 * e)
        else:
            e = 0.9 * (eps**(1/8))
            gz1 = switchfun(t, y + e * v_col)[0]
            gz2 = switchfun(t, y + 2 * e * v_col / 3)[0]
            gz3 = switchfun(t, y + e * v_col / 3)[0]
            gz4 = switchfun(t, y - e * v_col / 3)[0]
            gz5 = switchfun(t, y - 2 * e * v_col / 3)[0]
            gz6 = switchfun(t, y - e * v_col)[0]
            stats[9] += 6
            gxdit = (1 * gz1[ind] - 9 * gz2[ind] + 45 * gz3[ind] - 45 * gz4[ind] + 9 * gz5[ind] - 1 * gz6[ind]) / (20 * e)
        
        gxd[it] = gxdit
        
    return gxd, stats
#
#  End of graddif
#

def gradt(switchfun, t, y, ind, tol, stats):
    """
    Traducción de la función gradt de MATLAB a Python.
    Calcula el gradiente numérico según diferentes niveles de tolerancia.
    """
    # eps en MATLAB equivale a sys.float_info.epsilon en Python (~2.22e-16)
    eps = sys.float_info.epsilon 
    
    if tol > 1.e-7:
        e = 1.e-8
        g1 = switchfun(t, y)[0]
        g2 = switchfun(t + e, y)[0]
        stats[9] += 2  # stats(10) en MATLAB es stats[9] en Python
        gt = (g2[ind] - g1[ind]) / 1.e-8
        
    elif tol > 1.e-8:
        e = 1.e-6
        g1 = switchfun(t + e/2, y)[0]
        g2 = switchfun(t - e/2, y)[0]
        stats[9] += 2
        gt = (g1[ind] - g2[ind]) / e
        
    elif tol > 1.e-11:
        n = max(3, np.floor(15 + np.log10(tol)))
        e = 10**(-n)
        g1 = switchfun(t + e, y)[0]
        g1=np.atleast_1d(g1)
        g2,isterminal,direction= switchfun(t + e/2, y)
        g2=np.atleast_1d(g2)
        g3 = switchfun(t - e/2, y)[0]
        g3=np.atleast_1d(g3)
        g4 = switchfun(t - e, y)[0]
        g4=np.atleast_1d(g4)
        stats[9] += 4
        gt = (-g1[ind]/6 + 4*g2[ind]/3 - 4*g3[ind]/3 + g4[ind]/6) / e
        
    else:
        e = 0.9 * (eps ** (1/8))
        g1 = np.atleast_1d(switchfun(t + e, y))[0]
        g2 = switchfun(t + 2*e/3, y)[0]
        g3 = switchfun(t + e/3, y)[0]
        g4 = switchfun(t - e/3, y)[0]
        g5 = switchfun(t - 2*e/3, y)[0]
        g6 = switchfun(t - e, y)[0]
        stats[9] += 6
        gt = (1*g1[ind] - 9*g2[ind] + 45*g3[ind] - 45*g4[ind] + 9*g5[ind] - 1*g6[ind]) / (20 * e) 
    return gt, stats
#
#  End of gradt
#

# Funcion dummy

def dummy(t,y):
   yout=y.copy
   if Verbose >=0:
     print(f"\n\n Warning !!   No function actionatswitch provided \n")
     print(f"\n Check the options in the call to DISODESET \n\n");

   return yout

def classifypoint(x, y, inddisin, FUN, switchfun, stats, tol, rundata):
    # Asegurar que y sea un array de NumPy (columna o fila según corresponda)
    y = np.asarray(y)
    
    # Extraer variables de la estructura/objeto rundata
    Verbose = rundata.Verbose
    gradcomponents = rundata.gradcomponents
    exactgradient = rundata.exactgradient
    gradswitchfun = rundata.Gradient
    inddis=np.atleast_1d(inddisin)

    if Verbose >= 3:
        print(f"\nEnter classifypoint  {x} {inddis}")

    # Detectar si el punto pertenece a alguna superficie de conmutación
    g0a, _, direction = switchfun(x, y)
    g0=np.atleast_1d(g0a)
    direction=np.atleast_1d(direction)
    stats[9] += 1  # indexación 0 (originalmente stats(10))
    
    fout = FUN(x, y)
    stats[8] += 1  # indexación 0 (originalmente stats(9))
    
    # eps en Python se obtiene con np.finfo(float).eps
    eps_x = np.finfo(float).eps if x == 0 else np.finfo(x.dtype if hasattr(x, 'dtype') else float).eps
    eps_y = np.finfo(y.dtype).eps if hasattr(y, 'dtype') else np.finfo(float).eps
    
    paso = 1000 * max(100 * eps_x, max(np.atleast_1d(eps_y)) / max(0.001, np.linalg.norm(fout)))
    xxminus = x - paso
    yyminus = y - paso * fout
    gma, _, _ = switchfun(xxminus, yyminus)  # Punto antes de x,y
    gm=np.atleast_1d(gma)
    stats[9] += 1   
    xxplus = x + paso
    yyplus = y + paso * fout
    gpa, _, _ = switchfun(xxplus, yyplus)  # Punto después de x,y
    gp=np.atleast_1d(gpa)
 
    stats[9] += 1
    eps_1 = np.finfo(float).eps
 
    # Encontrar las superficies donde el punto conmuta
    # Ajuste de condiciones lógicas con NumPy
    condicion0=np.abs(g0) <= 500 * eps_1
    condicion1=(gp * gm ) <= 0
    condicion2=gp*direction >=0
    condicion = (condicion0 + condicion1)*condicion2
    
    ind = np.where(condicion >= 1 )[0]  # Devuelve índices basados en 0
  #  if np.all(gp * gm > 0) and np.all(np.abs(g0) > 500 * eps_1):
    if ind.size==0:
        xout = x
        yout = y
        gout = g0
        disctype = 0
        endslid = 0
        if Verbose >= 1:
            print(f"\n Point is not a discontinuity  {x} {y[0]} {np.min(np.abs(g0))}")
        indsliding = np.array([])
        return xout, yout, fout, gout, disctype, indsliding, endslid, stats

    if np.atleast_1d(inddis).size == 1 and np.all(inddis == 0):
        inddis = np.array([])
        
    ndis = ind.shape[0]
    ndisin = np.atleast_1d(inddis).size
    inddisout = np.zeros(ndis, dtype=int)
    
    ggt = np.zeros(ndis)
    ggxd = np.zeros((y.shape[0], ndis))
    nono = np.zeros(ndis)
    disctype = np.zeros(ndis)
    
    # Reordenar según el orden en que aparecieron

    for j in range(ndisin):
        i = np.where(ind == inddis[j])[0][0]
        inddisout[j] = ind[i]
        ind[i] = -1  # Usamos -1 en lugar de 0 para evitar conflictos con el índice 0 de Python
        
    # El resto de elementos que no eran -1
    ind_restantes = ind[ind != -1]
    inddisout[ndisin:ndis] = ind_restantes
    if ndis > 5:
        xout = x
        yout = y
        gout = g0
        disctype = -5
        endslid = np.zeros(ndis)
        indsliding = inddisout
        if Verbose >= 3:
            print(f"\n 4 or more discontinuities, x, y, indsliding  {x} {y[0]} {inddisout} \n")
        return xout, yout, fout, gout, disctype, indsliding, endslid, stats

    # Calcular los vectores normales a las superficies de conmutación
    for i in range(ndis):
        if exactgradient and gradcomponents[inddisout[i]] == 1:
            ggt[i], stats = gradt(switchfun, x, y, inddisout[i], min(1.e-8, tol / 10), stats)
            ggxd[:, i] = gradswitchfun(x, y, inddisout[i])
            stats[10] += 1  # originalmente stats(11)
        else:
            ggt[i], stats = gradt(switchfun, x, y, inddisout[i], min(1.e-8, tol / 10), stats)
            ggxd[:, i], stats = graddif(switchfun, x, y, inddisout[i], min(1.e-8, tol / 10), stats)
            
        nono[i] = np.sqrt(ggt[i]**2 + np.dot(ggxd[:, i], ggxd[:, i]))

    # Bifurcación según el número de discontinuidades detectadas
    if ndis == 1:
        xout, yout, fout, gout, disctype, indsliding, endslid, outputside, stats = onediscon(
            FUN, switchfun, x, y, xxminus, yyminus, xxplus, yyplus, tol,
            inddisout, g0, ggt[0], ggxd, nono, stats, rundata
        )

       
    elif ndis == 2:
        xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats = twodiscon(
            FUN, switchfun, x, y, tol, inddisout, g0, ggt, ggxd, nono, stats, rundata
        )
    elif ndis >= 3:
        inddispast = np.copy(inddisout)
        gpast = np.zeros(inddisout.shape)
        xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats = threediscon(
            FUN, switchfun, x, y, tol, gpast, inddispast, inddisout, g0, ggt, ggxd, nono, stats, rundata
        )
    return xout, yout, fout, gout, disctype, indsliding, endslid, stats
#
#  End of classifypoint
#

def disconflow(FUN, switchfun, x, y, xminus, xplus, yyminus, yyplus, inddis, gt, gfx, f1, f2, tol, stats, rundata):
    """
    Traducción de la función disconflow de MATLAB a Python.
    Analiza el flujo del campo vectorial a ambos lados de la superficie de conmutación.
    """
    eps = sys.float_info.epsilon

    # Extraer variables de configuración (asumiendo acceso por atributos)
    # Si rundata es un diccionario, cambia a: rundata['minfortangent1'], etc.
    minfortangent = [rundata.minfortangent1, rundata.minfortangent2]
    verbose = rundata.Verbose

    # Asegurar que las variables vectoriales sean arrays de numpy para álgebra lineal
    gfx = np.asarray(gfx)
    f1 = np.asarray(f1)
    f2 = np.asarray(f2)
    yminus = np.asarray(yyminus)
    yplus = np.asarray(yyplus)
    y = np.asarray(y)
    

    # Cálculo de normas y productos punto

    normgrad = np.sqrt(gt**2 +np.linalg.norm(gfx)**2)
    
    if abs(gt) < 1.e-14:
        normf1 = max(1+eps, np.linalg.norm(f1))
        normf2 = max(1+eps, np.linalg.norm(f2))
    else:
        normf1 = np.sqrt(1 + np.linalg.norm(f1)**2)
        normf2 = np.sqrt(1 + np.linalg.norm(f2)**2)

    gf1 = gt + gfx.T @ f1
    gf2 = gt + gfx.T @ f2
    
    gfxf1 = gf1 / max(eps, normgrad) / normf1
    gfxf2 = gf2 / max(eps, normgrad) / normf2

    # El bloque 'if/else' intermedio estaba comentado en el original de MATLAB
    ff1 = f1
    ff2 = f2

    if verbose >= 3:
        gminus = np.asarray(switchfun(xminus, yminus)[0])
        gplus = np.asarray(switchfun(xplus, yplus)[0])
        print(f"\n Data for classifying, x, inddis: {x:30.28g} {inddis}")
        print(f"\n Data for classifying, g: {gminus[inddis]} {gplus[inddis]}")
        print(f"\n Data for classifying, gt: {gt:22.20g}")
        print(f"\n Data for classifying, gfxf1: {gfxf1}")
        print(f"\n Data for classifying, gfxf2: {gfxf2}")
        print(f"\n Data for classifying, bound: {minfortangent[0]} {minfortangent[1]} {tol}")
        
  #  if x>7.6:
  #      input()

    # Inicialización de variables de retorno para evitar errores de referencia
    xout, yout, fout, gout = x, y.copy(), f1.copy(), np.zeros_like(f1)
    endslid = 0.0
    outputside = 0
    disctype = 3

    # --- Lógica de Clasificación de la Discontinuidad ---
    
    # Caso 1: Campo vectorial tangente a ambos lados
    if max(abs(gfxf1), abs(gfxf2)) < minfortangent[0]:
        if verbose >= 4:
            print("\n Warning! the vector field is tangent at both sides of the switching surface ")
            print(" The integration can not be reliable \n")
            
        
        endslid = max(gfxf1, gfxf2) - minfortangent[0]
        outputside = 0
        if np.linalg.norm(f2 - f1) < 150 * np.linalg.norm(yplus - yminus) + abs(xplus - xminus):
            endslid = 10 * eps
            if gfxf2 >= 0:
                outputside = 2
                xout = xplus
                yout = yplus.copy()
                gout = np.asarray(switchfun(xplus, yplus)[0])
            else:
                outputside = 1
                xout = xminus
                yout = yminus.copy()
                gout = np.asarray(switchfun(xminus, yminus)[0])

    # Caso 2: Sentidos opuestos y al menos uno alejado de cero
    elif (gfxf1 > -minfortangent[0]) and (gfxf2 < minfortangent[0]):
        endslid = max(-(gfxf1 + minfortangent[0]), gfxf2 - minfortangent[0])
        outputside = 0

    # Caso 3: Deslizamiento hacia el lado negativo (fuerte)
    elif (gfxf2 < minfortangent[0]) and (gfxf1 < -minfortangent[1]):
        yout = yminus.copy()
        xout = xminus
        outputside = 1
        endslid = abs(gfxf1) - minfortangent[0]

    # Caso 4: Deslizamiento hacia el lado negativo (débil)
    elif (gfxf2 < minfortangent[0]) and (gfxf1 < 0):
        stats[9] += 1  # stats(10) en MATLAB es stats[9] en Python
        xout = xminus
        yout = yminus.copy()
        fout = f1.copy()
        outputside = 1
        endslid = max(2 * eps, abs(gfxf1) - minfortangent[0])

    # Caso 5: Deslizamiento hacia el lado positivo (fuerte)
    elif (gfxf1 > -minfortangent[0]) and (gfxf2 > minfortangent[1]):
        yout = yplus.copy()
        xout = xplus
        outputside = 2
        endslid = abs(gfxf2) - minfortangent[0]

    # Caso 6: Deslizamiento hacia el lado positivo (débil)
    elif (gfxf1 > -minfortangent[0]) and (gfxf2 > 0):
        stats[9] += 1
        xout = xplus
        yout = yplus.copy()
        fout = f2.copy()
        outputside = 2
        endslid = max(2 * eps, abs(gfxf2) - minfortangent[0])

    # Caso 7: Dos salidas posibles del deslizamiento
    elif (gfxf2 > 0) and (gfxf1 < 0):
        print("\n  WARNING !!!!!! The solution has two possible exits from sliding")
        print("  The code chose the path most orthogonal to the switching manifold")
        print(f"\n Data for classifying, gfxf1: {gfxf1}")
        print(f"\n Data for classifying, gfxf2: {gfxf2}")
        print(f"\n Data for classifying, bound: {minfortangent[0]} {minfortangent[1]} {tol}")
        
        if abs(gfxf1 * normf1) > abs(gfxf2 * normf2):
            yout = yminus.copy()
            xout = xminus
            outputside = 1
            endslid = max(minfortangent[0] + 2 * eps, abs(gfxf1))
        else:
            outputside = 2
            yout = yplus.copy()
            xout = xplus
            endslid = max(minfortangent[0] + 2 * eps, abs(gfxf2))
            
    else:
        print(f"\n WARNING Unexpected case in classifying the discontinuity {gfxf1}  {gfxf2}")

    # --- Procesamiento de salidas finales ---
    if outputside == 1:
        fout = ff1.copy()
        gout = np.asarray(switchfun(xminus, yminus)[0])
        disctype = 1
    elif outputside == 2:
        fout = ff2.copy()
        gout = np.asarray(switchfun(xplus, yplus)[0])
        disctype = 1
    elif outputside == 0:
        yout = y.copy()
        xout = x
        gout = (np.asarray(switchfun(xminus, yminus)[0]) + np.asarray(switchfun(xplus, yplus)[0])) / 2
        gout=np.atleast_1d(gout)
        gout[inddis] = 0.0
        disctype = 3
        if abs(normf1 * gfxf1 - normf2 * gfxf2) < 1.e-15:
            fout = (f1 + f2) / 2
        elif abs(normf1 * gfxf1) < abs(normf2 * gfxf2):
            aux = (normf1 * gfxf1) / (normf2 * gfxf2)
            alfa = aux / (aux - 1)
            fout = (1 - alfa) * f1 + alfa * f2
        else:
            aux = (normf2 * gfxf2) / (normf1 * gfxf1)
            alfa = 1 / (1 - aux)
            fout = (1 - alfa) * f1 + alfa * f2

    if verbose >= 3:
        print(f"\n Discontinuity type: {disctype}")
    return xout, yout, fout, gout, endslid, disctype, outputside, gf1, gf2, stats

def onediscon(FUN, switchfun, x, y, xxminus, yyminus, xxplus, yyplus, tol, 
              inddisin, g0, ggtin, ggxd, nonoin, stats, rundata):
    """
    Traducción de la función onediscon de MATLAB a Python.
    """
    eps = sys.float_info.epsilon
    
    # Si rundata es un objeto o diccionario, puedes acceder a sus atributos así:
    # minfortangent = [rundata.minfortangent1, rundata.minfortangent2]  # O rundata['minfortangent1']
    # Verbose = rundata.Verbose
    # gradcomponents = rundata.gradcomponents
    # exactgradient = rundata.exactgradient
    # gradswitchfun = rundata.Gradient

    # Aseguramos que g0 sea un array de numpy para poder usar np.zeros_like
    inddis=inddisin[0]
    g0_arr = np.atleast_1d(np.asarray(g0))
    gminus = np.zeros_like(g0_arr)
    gplus = np.zeros_like(g0_arr)
    if np.isscalar(ggtin):
        ggt=ggtin 
    else:
        ggt=ggtin[0]
    if np.isscalar(nonoin):
        nono=nonoin 
    else:
        nono=nonoin[0]
    
    # Aseguramos que las variables de entrada que operan como vectores sean arrays de numpy
    y = np.asarray(y)
    ggxd = np.asarray(ggxd)

    if g0_arr[inddis] < 0:
        gminus = g0_arr.copy()
        xxminus = x
        yyminus = y.copy() if isinstance(y, np.ndarray) else y
    elif g0_arr[inddis] > 0:
        gplus = g0_arr.copy()
        xxplus = x
        yyplus = y.copy() if isinstance(y, np.ndarray) else y

    #
    # Compute points at both sides the switching surface if necessary
    #
   
    ii = 0
    paso1 = 0.0  # Inicializamos paso1 fuera del bucle por seguridad de ámbito
   
    while (gminus[inddis] >= 0) and (ii < 50):
        paso = (2**ii) * eps
        yyminus = y - paso * ggxd[:,0] / nono
        xxminus = x - paso * ggt / nono
        gminus = np.atleast_1d((switchfun(xxminus, yyminus)[0]))
        stats[9] += 1
        ii += 1
        paso1 = -paso

    ii = 0
   
    while (gplus[inddis] <= 0) and (ii < 50):
        paso = (2**ii) * eps
        yyplus = y + paso * ggxd[:,0] / nono
        xxplus = x + paso * ggt / nono
        gplus = np.atleast_1d(switchfun(xxplus, yyplus)[0])
        stats[9] += 1
        ii += 1
        paso1 = paso

    ii = 0
    paso1 = paso1 / 2
    paso = paso1
    
    while (paso1 > 1.e-19) and (ii <= 40):
        ii += 1
        paso1 = paso1 / 2
        xm = x + paso * ggt / nono
        ym = y + paso * ggxd[:,0] / nono
        gg = np.atleast_1d(switchfun(xm, ym)[0])
        stats[9] += 1
        
        if gg[inddis] < 0:
            xxminus = xm
            yyminus = ym
            paso = paso + paso1 / 2
        elif gg[inddis] > 0:
            xxplus = xm
            yyplus = ym
            paso = paso - paso1 / 2
        else:
            paso = paso - paso1 / 2

    # Evaluaciones de FUN
   
    f1 = FUN(xxminus, yyminus)
    f2 = FUN(xxplus, yyplus)
    stats[8] += 2  # stats(9) en MATLAB es stats[8] en Python
    

    # Llamada a la función externa disconflow
    # Nota: Asegúrate de que 'disconflow' ya esté traducida y disponible en tu entorno
   
    (xout, yout, fout, gout, endslid, disctype, outputside, _, _, _) = disconflow(
        FUN, switchfun, x, y, xxminus, xxplus, yyminus, 
        yyplus, inddis, ggt, ggxd, f1, f2, tol, stats, rundata
    )
    indsliding = inddis
   
    
    return xout, yout, fout, gout, disctype, indsliding, endslid, outputside, stats
#
#  End of onediscon
#

def twodiscon(FUN, switchfun, x, y, tol, inddis, g0, ggt, ggxd, nono, stats, rundata):
    # Inicializaciones básicas
    Verbose = getattr(rundata, 'Verbose', 0)
    ndis = 2
    endslid = np.zeros(2)
    
    # En Python, para vectores 1D, el producto escalar se hace con np.dot o @
    # ggt(1)'*ggt(2) + ggxd(:,1)'*ggxd(:,2)
    paralel = ggt[0] * ggt[1] + np.dot(ggxd[:, 0], ggxd[:, 1])
    paralel = paralel / (nono[0] * nono[1])
    
    gout = np.copy(g0)
    # Convertimos los índices de MATLAB (base 1) a Python (base 0)
    
    idis1 = int(inddis[0])
    idis2 = int(inddis[1])
    
    if abs(abs(paralel) - 1.0) < 1.e-8:
        xout = x
        yout = np.copy(y)
        fout = 0.0
        gout = np.copy(g0)
        disctype = -7
        indsliding = np.copy(inddis)
        outputsideout = np.array([0, 0])
        if Verbose >= 3:
            print(f"\n Tangent surfaces {x} {min(abs(g0))} {indsliding}")
        return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats

    # Operación elemento a elemento de MATLAB (./)
    den = (np.outer(ggt, ggt) + np.dot(ggxd.T, ggxd)) / nono
    
    x11, x12, x21, x22 = 0, 0, 0, 0
    y11, y12, y21, y22 = None, None, None, None
    
    if g0[idis1] < 0 and g0[idis2] < 0:
        x11, y11 = x, np.copy(y)
    elif g0[idis1] < 0 and g0[idis2] > 0:
        x12, y12 = x, np.copy(y)
    elif g0[idis1] > 0 and g0[idis2] < 0:
        x21, y21 = x, np.copy(y)
    elif g0[idis1] > 0 and g0[idis2] > 0:
        x22, y22 = x, np.copy(y)
        
    dete = den[1, 1] * den[0, 0] - den[0, 1] * den[1, 0]
    
    # Bucle de aproximación para x11
    if x11 == 0:
        aa = -3.5e-13
        g11 = np.copy(g0)
        ii = 1
        # inddis(1:2) en MATLAB son los dos primeros índices. Restamos 1 para evaluar en Python.
        idx_eval = (inddis[:2]).astype(int)
        while np.any(g11[idx_eval] >= 0) and ii < 20:
            c1 = ((den[1, 1] - den[0, 1]) * aa * ii - (den[1, 1] * g0[idis1] - den[0, 1] * g0[idis2])) / dete
            c2 = ((den[0, 0] - den[1, 0]) * aa * ii - (den[0, 0] * g0[idis2] - den[1, 0] * g0[idis1])) / dete
            x11 = x + c1 * ggt[0] / nono[0] + c2 * ggt[1] / nono[0]
            y11 = y + c1 * ggxd[:, 0] / nono[0] + c2 * ggxd[:, 1] / nono[1]
            g11, isterminal, direction = switchfun(x11, y11)
            stats[9] += 1  # stats(10) en MATLAB -> índice 9 en Python
            ii += 1
        if np.any(g11[idx_eval] >= 0):
            print('\n Falta x11 ', g11[idx_eval])
            input("Press Enter to continue...") # Emula el 'pause' de MATLAB

    # Bucle de aproximación para x12
    if x12 == 0:
        aa = 3.e-13
        g12 = np.copy(g0)
        ii = 1
        while (g12[idis1] >= 0 or g12[idis2] <= 0) and ii < 15:
            c1 = ((-den[1, 1] - den[0, 1]) * aa * ii - (den[1, 1] * g0[idis1] - den[0, 1] * g0[idis2])) / dete
            c2 = ((den[0, 0] + den[1, 0]) * aa * ii - (den[0, 0] * g0[idis2] - den[1, 0] * g0[idis1])) / dete
            x12 = x + c1 * ggt[0] / nono[0] + c2 * ggt[1] / nono[1]
            y12 = y + c1 * ggxd[:, 0] / nono[0] + c2 * ggxd[:, 1] / nono[1]
            g12, isterminal, direction = switchfun(x12, y12)
            stats[9] += 1
            ii += 1
        if g12[idis1] >= 0 or g12[idis2] <= 0:
            print('\n Falta x12 ', g12[idx_eval])
            input("Press Enter to continue...")

    # Bucle de aproximación para x21
    if x21 == 0:
        aa = 5.e-13
        g21 = np.copy(g0)
        ii = 1
        while (g21[idis1] <= 0 or g21[idis2] >= 0) and ii < 15:
            c1 = ((den[1, 1] + den[0, 1]) * aa - (den[1, 1] * g0[idis1] - den[0, 1] * g0[idis2])) / dete
            c2 = ((-den[0, 0] - den[1, 0]) * aa - (den[0, 0] * g0[idis2] - den[1, 0] * g0[idis1])) / dete
            x21 = x + c1 * ggt[0] / nono[0] + c2 * ggt[1] / nono[1]
            y21 = y + c1 * ggxd[:, 0] / nono[0] + c2 * ggxd[:, 1] / nono[1]
            g21, isterminal, direction = switchfun(x21, y21)
            stats[9] += 1
            ii += 1
        if g21[idis1] <= 0 or g21[idis2] >= 0:
            print('\n Falta x21 ') # Nota: El código original de MATLAB usa 'inddisout' que no está definido, se asume inddis
            input("Press Enter to continue...")

    # Bucle de aproximación para x22
    if x22 == 0:
        aa = 3.5e-13
        g22 = np.copy(g0)
        ii = 1
        idx_eval = (inddis[:2]).astype(int)
        while np.any(g22[idx_eval] <= 0) and ii < 15:
            c1 = ((den[1, 1] - den[0, 1]) * aa * ii - (den[1, 1] * g0[idis1] - den[0, 1] * g0[idis2])) / dete
            c2 = ((den[0, 0] - den[1, 0]) * aa * ii - (den[0, 0] * g0[idis2] - den[1, 0] * g0[idis1])) / dete
            x22 = x + c1 * ggt[0] / nono[0] + c2 * ggt[1] / nono[1]
            y22 = y + c1 * ggxd[:, 0] / nono[0] + c2 * ggxd[:, 1] / nono[1]
            g22, isterminal, direction = switchfun(x22, y22)
            stats[9] += 1
            ii += 1
        if np.any(g22[idx_eval] <= 0):
            print('\n Falta x22 ', g22[idx_eval])
            input("Press Enter to continue...")

    if ndis >= 2 and Verbose >= 3:
        print(f"\n {ndis} discontinuities at this point: {ndis} {inddis}")

    # Inicialización de matrices para flujos
    y_len = y.shape[0] if hasattr(y, 'shape') else len(y)
    f1 = np.zeros((y_len, ndis))
    f2 = np.zeros((y_len, ndis))
    outputside = np.zeros((ndis, ndis))
    outputsideout = np.array([0, 0])
    
    f1[:, 0] = FUN(x11, y11)
    f2[:, 0] = FUN(x21, y21)
    f2[:, 1] = FUN(x22, y22)
    f1[:, 1] = FUN(x12, y12)
    stats[8] += 4  # stats(9) -> índice 8
    
    # Descomposición de llamadas externas a `disconflow`
    # Se asume que disconflow devuelve una tupla/lista con sus correspondientes valores descritos en MATLAB
    xout11, yout11, fout11, gout11, endslid11, _, outputside[0, 0], gf11a, gf11b, stats = disconflow(FUN, switchfun, x, y, x11, x21, y11, y21, inddis[0], ggt[0], ggxd[:, 0], f1[:, 0], f2[:, 0], tol, stats, rundata)
    xout12, yout12, fout12, gout12, endslid12, _, outputside[0, 1], gf12a, gf12b, stats = disconflow(FUN, switchfun, x, y, x12, x22, y12, y22, inddis[0], ggt[0], ggxd[:, 0], f1[:, 1], f2[:, 1], tol, stats, rundata)
    xout21, yout21, fout21, gout21, endslid21, _, outputside[1, 0], gf21a, gf21b, stats = disconflow(FUN, switchfun, x, y, x11, x12, y11, y12, inddis[1], ggt[1], ggxd[:, 1], f1[:, 0], f1[:, 1], tol, stats, rundata)
    xout22, yout22, fout22, gout22, endslid22, _, outputside[1, 1], gf22a, gf22b, stats = disconflow(FUN, switchfun, x, y, x21, x22, y21, y22, inddis[1], ggt[1], ggxd[:, 1], f2[:, 0], f2[:, 1], tol, stats, rundata)

    if Verbose >= 3:
        gg11 = switchfun(x11, y11)[0][idx_eval]
        gg12 = switchfun(x12, y12)[0][idx_eval]
        gg21 = switchfun(x21, y21)[0][idx_eval]
        gg22 = switchfun(x22, y22)[0][idx_eval]
        stats[9] += 4
        if gg11[0] > 0 or gg11[1] > 0 or gg12[0] > 0 or gg12[1] < 0 or gg21[0] < 0 or gg21[1] > 0 or gg22[0] < 0 or gg22[1] < 0:
            print(f'\n 1 cuatro ges {gg11[0]} {gg12[0]} {gg21[0]} {gg22[0]} ')
            print(f'\n 2 cuatro ges {gg11[1]} {gg12[1]} {gg21[1]} {gg22[1]} ')
            print(f'\n discontinuities {inddis[0]} {inddis[1]}')

    if endslid11 > 0 and endslid12 > 0 and outputside[0, 0] == outputside[0, 1]:
        endslid[0] = min(endslid11, endslid12)
    elif endslid11 < 0 and endslid12 < 0:
        endslid[0] = max(endslid11, endslid12)

    if endslid21 > 0 and endslid22 > 0 and outputside[1, 0] == outputside[1, 1]:
        endslid[1] = min(endslid21, endslid22)
    elif endslid21 < 0 and endslid22 < 0:
        endslid[1] = max(endslid21, endslid22)

    if Verbose >= 3:
        print(f'\n inddisout {inddis[0]} {inddis[1]} ')
        print(f'\n endslid {endslid11} {endslid12} {endslid21} {endslid22} ')
        print(f'\n outputside {outputside[0,0]} {outputside[0,1]} {outputside[1,0]} {outputside[1,1]} ')

    # eps en MATLAB para un número escalar es np.finfo(float).eps
    # eps(x) es la distancia al siguiente número en coma flotante.
 #   paso = max(1000 * np.spacing(x), tol / 1000)
    cond11a = (outputside[0, 0] == 0 and outputside[1, 0] == 0 and outputside[0, 1] == 1)
    cond11b = (outputside[0, 0] == 0 and outputside[1, 0] == 2 and outputside[1, 1] == 1)

    # Lógica de bifurcaciones de deslizamientos y retornos
    if endslid[0] > 0 and endslid[1] > 0:
        if outputside[1, 0] == 1:
            fout, xout, yout, gout = fout11, xout11, yout11, gout11
            outputsideout = np.array([1, 1])
        else:
            fout, xout, yout, gout = fout12, xout12, yout12, gout12
            outputsideout = np.array([1, 2])
        indsliding = 0
        disctype = 1
        return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats

    elif endslid[0] < 0 and endslid[1] < 0:
        normgrad = np.sqrt(ggt[1]**2 + np.dot(ggxd[:, 1], ggxd[:, 1]))
        gfxf1 = (ggt[1] + np.dot(ggxd[:, 1], fout11)) / max(np.spacing(1.0), normgrad)
        gfxf2 = (ggt[1] + np.dot(ggxd[:, 1], fout12)) / max(np.spacing(1.0), normgrad)
        alfa = gfxf1 / (gfxf1 - gfxf2)
        fout = (1.0 - alfa) * fout11 + alfa * fout12
        disctype = 3
        xout, yout = x, np.copy(y)
        indsliding = np.copy(inddis)
        outputsideout = np.array([0, 0])
        return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats

    elif endslid[0] < 0:
        normgrad = np.sqrt(ggt[1]**2 + np.dot(ggxd[:, 1], ggxd[:, 1]))
        gfxf1 = (ggt[1] + np.dot(ggxd[:, 1], fout11)) / max(np.spacing(1.0), normgrad)
        gfxf2 = (ggt[1] + np.dot(ggxd[:, 1], fout12)) / max(np.spacing(1.0), normgrad)
        if gfxf1 * gfxf2 < 0:
            disctype = 3
            xout, yout = x, np.copy(y)
            indsliding = np.copy(inddis)
            endslid[1] = endslid[0]
            alfa = gfxf1 / (gfxf1 - gfxf2)
            fout = (1.0 - alfa) * fout11 + alfa * fout12
            outputsideout = np.array([0, 0])
            return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats
        else:
            disctype = 3
            if gfxf1 < 0:
                fout, xout, yout, gout = fout11, xout11, yout11, gout11
                outputsideout = np.array([0, 1])
            else:
                fout, xout, yout, gout = fout12, x12, y12, gout12
                outputsideout = np.array([0, 2])
            indsliding = inddis[0]
            return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats

    elif endslid[1] < 0:
        normgrad = np.sqrt(ggt[0]**2 + np.dot(ggxd[:, 0], ggxd[:, 0]))
        gfxf1 = (ggt[0] + np.dot(ggxd[:, 0], fout21)) / max(np.spacing(1.0), normgrad)
        gfxf2 = (ggt[0] + np.dot(ggxd[:, 0], fout22)) / max(np.spacing(1.0), normgrad)
        if gfxf1 * gfxf2 < 0:
            disctype = 3
            alfa = gfxf1 / (gfxf1 - gfxf2)
            fout = (1.0 - alfa) * fout21 + alfa * fout22
            xout, yout = x, np.copy(y)
            indsliding = np.copy(inddis)
            endslid[0] = endslid[1]
            outputsideout = np.array([0, 0])
            return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats
        else:
            disctype = 3
            if gfxf1 < 0:
                fout, xout, yout, gout = fout21, xout21, yout21, gout21
                outputsideout = np.array([1, 0])
            else:
                fout, xout, yout, gout = fout22, xout22, yout22, gout22
                endslid[0] = gfxf2
                outputsideout = np.array([2, 0])
            indsliding = inddis[1]
            return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats

    elif endslid[0] > 0:
        if outputside[0, 0] == 1:
            fout, xout, yout = fout21, xout21, yout21
            if outputside[1, 0] == 0:
                indsliding = inddis[1]
                endslid[1] = endslid21
                disctype = 3
                outputsideout = np.array([1, 0])
            else:
                disctype = 1
                indsliding = []
                outputsideout = np.array([1, int(outputside[1, 0])])
            return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats
        elif outputside[0, 0] == 2:
            fout, xout, yout = fout22, xout22, yout22
            if outputside[1, 1] == 0:
                indsliding = inddis[1]
                endslid[1] = endslid22
                disctype = 3
                outputsideout = np.array([2, 0])
            else:
                disctype = 1
                indsliding = []
                outputsideout = np.array([int(outputside[1, 1]), 2])
            return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats

    elif endslid[1] > 0:
        if outputside[1, 0] == 1:
            fout, xout, yout = fout11, xout11, yout11
            if outputside[0, 0] == 0:
                indsliding = inddis[0]
                endslid[0] = endslid11
                disctype = 3
                outputsideout = np.array([0, 1])
            else:
                disctype = 1
                indsliding = []
                outputsideout = np.array([int(outputside[0, 0]), 1])
            return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats
        elif outputside[1, 0] == 2:
            fout, xout, yout = fout22, xout22, yout22
            if outputside[1, 1] == 0:
                indsliding = inddis[1]
                endslid[0] = endslid22
                disctype = 3
                outputsideout = np.array([0, 2])
            else:
                disctype = 1
                indsliding = []
                outputsideout = np.array([2, int(outputside[1, 1])])
            return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats

    elif cond11a or cond11b:
        xout, yout, fout, gout = xout11, yout11, fout11, gout11
        indsliding = []
        disctype = 1
        endslid[0] = endslid11
        endslid[1] = endslid[0]
        outputsideout = np.array([1, 1])
        return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats

    elif outputside[0, 0] == 2 and outputside[0, 1] == 0 and outputside[1, 0] == 2 and outputside[1, 1] == 0:
        normgrad = np.sqrt(ggt[0]**2 + np.dot(ggxd[:, 0], ggxd[:, 0]))
        gfxf1 = (ggt[0] + np.dot(ggxd[:, 0], fout22)) / max(np.spacing(1.0), normgrad)
        gfxf2 = (ggt[1] + np.dot(ggxd[:, 1], fout12)) / max(np.spacing(1.0), normgrad)
        
        if gfxf1 > 0 and gfxf2 > 0:
            outputsideout = np.array([2, 0])
            fout, xout, yout = fout22, x22, y22
            indsliding = inddis[1]
            endslid = np.array([endslid11, endslid22])
        elif gfxf1 > 0:
            outputsideout = np.array([2, 0])
            fout, xout, yout = fout22, x22, y22
            indsliding = inddis[1]
            endslid = np.array([endslid21, endslid22])
        elif gfxf2 > 0:
            outputsideout = np.array([0, 2])
            fout, xout, yout = fout12, x12, y12
            indsliding = inddis[0]
            endslid = np.array([endslid11, endslid12])
        else:
            # Resolución matricial/algebraica para Co-dimensión 2
            denom_a = (-gf11b*gf21b + gf12b*gf21b + gf12a*gf22a - gf12b*gf22a + gf11b*gf22b - gf12a*gf22b)
            a2 = -((-gf12b*gf21b + gf12a*gf22b) / denom_a)
            a3 = -((gf12b*gf22a - gf11b*gf22b) / denom_a)
            a4 = -((gf11b*gf21b - gf12a*gf22a) / denom_a)
            fout = a2 * f2[:, 0] + a3 * f1[:, 1] + a4 * f2[:, 1]
            xout, yout = x, np.copy(y)
            indsliding = np.copy(inddis)
            endslid = np.array([endslid12, endslid22])
            outputsideout = np.array([0, 0])
            
        disctype = 3
        return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats

    elif outputside[0, 0] == 0 and outputside[0, 1] == 2 and outputside[1, 0] == 0 and outputside[1, 1] == 2:
        outputsideout = np.array([2, 2])
        fout, xout, yout = fout22, x22, y22
        indsliding = []
        endslid = np.array([endslid12, endslid22])
        disctype = 0
        return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats
    elif outputside[0, 0] == 1 and outputside[0, 1] == 0 and outputside[1, 0] == 1 and outputside[1, 1] == 0:
       normgrad = np.sqrt(ggt[0]**2 + np.dot(ggxd[:, 0], ggxd[:, 0]))
       gfxf1 = (ggt[0] + np.dot(ggxd[:, 0], fout22)) / max(np.spacing(1.0), normgrad)
       gfxf2 = (ggt[1] + np.dot(ggxd[:, 1], fout12)) / max(np.spacing(1.0), normgrad)
 
       disctype = 3
       alfa = gfxf1 / (gfxf1 - gfxf2)
       fout = (1.0 - alfa) * fout21 + alfa * fout22
       xout, yout = x, np.copy(y)
       indsliding = np.copy(inddis)
       endslid[0] = endslid[1]
       outputsideout = np.array([0, 0])
       return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats
    # Caso por defecto si no entra en ningún condicional anterior
    print(f'\n Caso no contemplado en twodiscon at {x} \n')
    print(f'\n inddisout {inddis[0]} {inddis[1]} ')
    print(f'\n endslid {endslid11} {endslid12} {endslid21} {endslid22} ')
    print(f'\n outputside {outputside[0,0]} {outputside[0,1]} {outputside[1,0]} {outputside[1,1]} ')
    xout = x
    yout = np.copy(y)
    fout = fout11
    disctype = -5
    indsliding = np.copy(inddis)
    
    return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats
#
#  End of twodiscon
#

def threediscon(FUN, switchfun, x, y, tol, gpast, inddispast, inddis, g0, ggt, ggxd, nono, stats, rundata):
#    minfortangent = [rundata.minfortangent1, rundata.minfortangent2]
    Verbose = rundata.Verbose
 #   gradswitchfun = rundata.Gradient

    # Aseguramos que inddis sea un array de NumPy unidimensional
    inddis = np.atleast_1d(inddis)
    ndis = inddis.shape[0]
#    search = True
    indsal = ndis - 1  # Ajuste base 0
    indfirst1 = 0
    indfirst2 = 1
#    probar = False

    order = np.arange(ndis)
    orderinv = order.copy()
    orderred = order[:ndis - 1]

    # Constante épsilon de punto flotante de 64 bits (equivalente a eps(1) en MATLAB)
    eps_val = np.finfo(float).eps

    for veces in range(1, 3):  # Equivalente a 1:2 en MATLAB (ejecuta para veces = 1 y 2)
        indsal = order[ndis - 1]
        index = inddis[order[ndis - 1]]

        gminus = np.zeros_like(g0, dtype=float)
        gplus = np.zeros_like(g0, dtype=float)
        endslid = np.zeros(ndis, dtype=float)

        xxminus = x
        yyminus = y
        xxplus = x
        yyplus = y

        if g0[index] < 0:
            gminus = g0.copy()
            xxminus = x
            yyminus = y
        elif g0[index] > 0:
            gplus = g0.copy()
            xxplus = x
            yyplus = y

        # Compute points at both sides of the switching surface if necessary
        ii = 0
        paso1 = 0.0
        while (gminus[index] >= 0) and (ii < 50):
            paso = (2**ii) * eps_val
            yyminus = y - paso * ggxd[:, indsal] / nono[indsal]
            xxminus = x - paso * ggt[indsal] / nono[indsal]
            gminus, _, _ = switchfun(xxminus, yyminus)
            stats[9] += 1  # stats(10) en MATLAB -> stats[9] en Python
            ii += 1
            paso1 = -paso

        if ii >= 50:
            print('\n Warning!!! xxminus not reached')

        ii = 0
        while (gplus[index] <= 0) and (ii < 50):
            paso = (2**ii) * eps_val
            yyplus = y + paso * ggxd[:, indsal] / nono[indsal]
            xxplus = x + paso * ggt[indsal] / nono[indsal]
            gplus, _, _ = switchfun(xxplus, yyplus)
            stats[9] += 1
            ii += 1
            paso1 = paso

        if ii >= 50:
            print('\n Warning!!! xxplus not reached')

        ii = 0
        paso1 = paso1 / 2.0
        paso = paso1

        while (paso1 > 1.e-19) and (ii <= 40):
            ii += 1
            paso1 = paso1 / 2.0
            xm = x + paso * ggt[indsal] / nono[indsal]
            ym = y + paso * ggxd[:, indsal] / nono[indsal]
            gg, _, _ = switchfun(xm, ym)
            stats[9] += 1

            if gg[index] < 0:
                xxminus = xm
                yyminus = ym
                paso = paso + paso1 / 2.0
            elif gg[index] > 0:
                xxplus = xm
                yyplus = ym
                paso = paso - paso1 / 2.0
            else:
                paso = paso - paso1 / 2.0

        # Asumimos que la función projnew devuelve los valores mapeados correctamente
        xminus, yminus, gminus, _, direction, error, _, stats = projnew(
            switchfun, inddis[orderred], xxminus, yyminus,
            ggt[orderred], ggxd[:, orderred], np.zeros(ndis - 1), stats, Verbose
        )

        indi_arr = np.where(inddispast == inddis[indsal])[0]
        gpast1 = gpast.copy()
        if len(indi_arr) > 0:
            gpast1[indi_arr[0]] = -1
            

        xplus, yplus, gplus, _, direction, error, _, stats = projnew(
            switchfun, inddis[orderred], xxplus, yyplus,
            ggt[orderred], ggxd[:, orderred], np.zeros(ndis - 1), stats, Verbose
        )

        gpast2 = gpast.copy()
        if len(indi_arr) > 0:
            gpast2[indi_arr[0]] = 1

        if np.any(gplus[inddispast].T * gpast < 0):
            print(' fallo', veces, ndis, inddispast, gpast2, gplus[inddispast], gpast)
            input("Press Enter to continue...")  # Equivalente a 'pause'

        if np.any(gminus[inddispast[ndis:]].T * gpast1[ndis:] < 0):
            print(' fallo', veces, ndis, inddispast, gpast1[ndis:], gminus[inddispast[ndis:]], gpast)

        if ndis == 3:
            xoutm, youtm, f1, goutm, disctype1, indsliding1, endslid1, outputside1, stats = twodiscon(
                FUN, switchfun, xminus, yminus, tol, inddis[orderred],
                gminus, ggt[orderred], ggxd[:, orderred],
                nono[orderred], stats, rundata
            )
            xoutp, youtp, f2, goutp, disctype2, indsliding2, endslid2, outputside2, stats = twodiscon(
                FUN, switchfun, xplus, yplus, tol, inddis[orderred],
                gplus, ggt[orderred], ggxd[:, orderred],
                nono[orderred], stats, rundata
            )
        else:
            xoutm, youtm, f1, goutm, disctype1, indsliding1, endslid1, outputside1, stats = threediscon(
                FUN, switchfun, xminus, yminus, tol, gpast1, inddispast, inddis[orderred],
                gminus, ggt[orderred], ggxd[:, orderred], nono[orderred], stats, rundata
            )
            xoutp, youtp, f2, goutp, disctype2, indsliding2, endslid2, outputside2, stats = threediscon(
                FUN, switchfun, xplus, yplus, tol, gpast2, inddispast, inddis[orderred],
                gplus, ggt[orderred], ggxd[:, orderred], nono[orderred], stats, rundata
            )

        if disctype1 == -7 or disctype2 == -7:
            xout = x
            yout = y
            fout = 0
            gout = g0
            disctype = -7
            indsliding = inddis
            outputsideout = np.append(outputside2, 0)
            return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats

        xout, yout, fout, gout, endslid0, disctype, outputside, gf1, gf2, stats = disconflow(
            FUN, switchfun, x, y, xoutm, xoutp, youtm, youtp,
            inddis[indsal], ggt[indsal], ggxd[:, indsal], f1, f2, tol, stats, rundata
        )

        if (disctype1 == -5 and outputside != 2) or (disctype2 == -5 and outputside != 1):
            xout = x
            yout = y
            fout = f1
            gout = g0
            disctype = -5
            indsliding = inddis
            endslid = np.append(endslid2, 1.e-10)
            outputsideout = np.append(outputside1, outputside)
            return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats

        if outputside == 0 and np.all(outputside1 == 0) and np.all(outputside2 == 0):
            indsliding = inddis
            endslid = np.concatenate([endslid2[:indsal], [endslid0], endslid2[indsal:]])
            endslid = np.append(endslid2, endslid0)
            endslid = endslid[orderinv]
            outputsideout = np.concatenate([outputside2[:indsal], [outputside], outputside2[indsal:]])
            disctype = 3
            return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats

        elif outputside != 0:
            if outputside == 1:
                indsliding = indsliding1
                endslid = np.append(endslid1, endslid0)[orderinv]
                outputsideout = np.append(outputside1, outputside)[orderinv]
                gaux = gout[inddis[indsal]]
                gout = goutm.copy()
                gout[inddis[indsal]] = gaux
                disctype = disctype1
            elif outputside == 2:
                indsliding = indsliding2
                endslid = np.append(endslid2, endslid0)[orderinv]
                outputsideout = np.append(outputside2, outputside)[orderinv]
                disctype = disctype2
                gaux = gout[inddis[indsal]]
                gout = goutp.copy()
                gout[inddis[indsal]] = gaux
            else:
                print(' raro raro')
                indsliding = indsliding2
                endslid = np.append(endslid2, endslid0)
                outputsideout = np.append(outputside2, outputside)
                disctype = 3
            return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats

        elif np.all((outputside1 - outputside2) == 0):
            gaux = gout[inddis[indsal]]
            gout = goutm.copy()
            gout[inddis[indsal]] = gaux
            outputsideout = np.append(outputside2, outputside)[orderinv]
            indsal = np.where(outputsideout != 0)[0]
            ndissal = indsal.shape[0]
            index = inddis[indsal]
            endslid = np.append(endslid2, endslid0)[orderinv]
            indsliding = inddis.copy()
            
            # Eliminación de elementos
            indsliding = np.delete(indsliding, indsal)
            ggt = np.delete(ggt, indsal)
            ggxd = np.delete(ggxd, indsal, axis=1)
            nono = np.delete(nono, indsal)

            if ndissal == ndis:
                disctype = 1
            else:
                disctype = 3
            return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats

        elif np.any((outputside1 - outputside2) != 0):
#            endslid1a = np.append(endslid1, endslid0)[orderinv]
#            endslid2a = np.append(endslid2, endslid0)[orderinv]
            outputside1a = np.append(outputside1, endslid0)[orderinv]
            outputside2a = np.append(outputside2, endslid0)[orderinv]
 #           indsala = indsal
            
            indsal = np.where((outputside1a - outputside2a) != 0)[0]
            ndissal = indsal.shape[0]

            if ndissal == 1 and veces <= 2:
                # Construcción del vector order
                i_val = indsal[0]
                order = np.concatenate([np.arange(i_val), np.arange(i_val + 1, ndis), [i_val]])
                orderred = order[:ndis - 1]
                orderinv = np.arange(ndis)
                orderinv[i_val] = ndis - 1
                orderinv[i_val + 1:ndis] = np.arange(i_val, ndis - 1)

            if ndissal >= 2 and veces <= 2:
                indfirst1 = indsal[0]
                indfirst2 = indsal[1]
                aux = np.arange(ndis)
                ini = np.where((aux == indfirst1) | (aux == indfirst2))[0]
                aux = np.delete(aux, ini)
                indsal_val = np.min(aux)
                ind = np.array([indfirst1, indfirst2, indsal_val])
                ind1 = np.min(ind)
                ind3 = np.max(ind)
                ll = np.where((ind != ind1) & (ind != ind3))[0]
                ind2 = ind[ll[0]]

                order = np.concatenate([
                    [indfirst1, indfirst2],
                    np.arange(ind1),
                    np.arange(ind1 + 1, ind2),
                    np.arange(ind2 + 1, ind3),
                    np.arange(ind3 + 1, ndis),
                    [indsal_val]
                ])
                orderred = order[:ndis - 1]
                orderinv = np.arange(ndis)
                orderinv[indfirst1] = 0
                orderinv[indfirst2] = 1

                if ind1 == 0 and ind2 == 1:
                    orderinv[2:ind3] = np.arange(2, ind3)
                    orderinv[ind3 + 1:ndis] = np.arange(ind3, ndis - 1)
                elif ind1 == 0:
                    orderinv[1:ind2] = np.arange(2, ind2 + 1)
                    orderinv[ind2 + 1:ind3] = np.arange(ind2 + 1, ind3)
                    orderinv[ind3 + 1:ndis] = np.arange(ind3, ndis - 1)
                else:
                    orderinv[:ind1] = np.arange(2, 2 + ind1)
                    orderinv[ind1 + 1:ind2] = np.arange(3 + ind1, ind2 + 2)
                    orderinv[ind2 + 1:ind3] = np.arange(4 + ind2, ind3 + 3)
                    orderinv[ind3 + 1:ndis] = np.arange(ind3, ndis - 1)

                orderinv[indsal_val] = ndis - 1

            if ndissal > 2 or veces >= 3:
                order = order[order]
                orderred = order[:ndis - 1]
                orderinv = orderinv[orderinv]
                ndissal = 1

            if np.any((order[orderinv] - np.arange(ndis)) != 0):
                print(indfirst1, indfirst2, indsal, ind1, ind2, ind2, ' mal order ', order, orderinv)
                input("Press Enter to continue...")

            if ndissal > 2:
                print(f'\n ndis1, indsal, {ndis} {indsal} {outputside1} {outputside}')
                print(f'\n ndis2, indsal, {ndis} {indsal} {outputside2} {outputside}')
                xout = x
                yout = y
                gout = g0.copy()
                fout = FUN(x, y)
                disctype = -5
                endslid = -np.ones(ndis, dtype=float)
                outputsideout = np.zeros(ndis, dtype=float)
                indsliding = inddis.copy()
                return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats

            indsal = indsal[0]
        else:
            disctype = 3
            endslid = np.append(endslid2, endslid0)
            outputsideout = np.append(outputside2, outputside)
            indsliding = inddis.copy()
            indsliding = np.delete(indsliding, indsal)
            return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats

    xout = x
    yout = y
    gout = g0.copy()
    fout = FUN(x, y)
    disctype = -5

    i_sal = indsal if np.isscalar(indsal) else indsal[0]
    endslid = np.concatenate([endslid1[:i_sal], [endslid0], endslid1[i_sal:]])
    outputsideout = np.zeros(ndis, dtype=float)
    endslid[i_sal] = -1.e-10
    outputsideout[i_sal] = 0
    indsliding = inddis.copy()

    if Verbose >= 3:
        # Nota: Asume existencia previa de la variable 'inddisout' en el scope global si no es argumento
        print(f'\n 3 discontinuities, x, y, indsliding {x} {y[0]} \n')

    return xout, yout, fout, gout, disctype, indsliding, endslid, outputsideout, stats
#
#  End of threediscon
#

def FindDiscpro(FUN, switchfun, H, X, Y, WRK, xx, yy, xdis, tdis, ydis, idis, indsliding,
                 nout, npoints, stats, tol, rundata):
    """
    Traducción de la función FindDiscpro de MATLAB a Python.
    Requiere que 'estirapro', 'gradt', 'graddif', 'proj' y las funciones 
    dentro de 'rundata' estén implementadas adecuadamente en Python.
    """
    
  # Extraer variables desde el objeto rundata de forma segura utilizando getattr
 #   minfortangent = [getattr(rundata, 'minfortangent1', 0), getattr(rundata, 'minfortangent2', 0)]
    Verbose = getattr(rundata, 'Verbose', 0)
    gradcomponents = getattr(rundata, 'gradcomponents', [])
    xend = getattr(rundata, 'Xend', 0)
    tspan = getattr(rundata, 'tspan', [])
    exactgradient = getattr(rundata, 'exactgradient', False)
    doatswitch = getattr(rundata, 'ActionSwitch', None)
    gradswitchfun = getattr(rundata, 'Gradient', None)
    Refine = getattr(rundata, 'Refine', 0)
    solyes = getattr(rundata, 'nargout', 0)
    
    WRKout = np.copy(WRK)
    Hout = max(H, 1.e-9)
    inddis = np.atleast_1d(indsliding) #  sliding surfaces
    ndis = len(inddis)
    wt = np.zeros(ndis)
    
    Y = np.atleast_1d(Y)
    neq = Y.shape[0]
    w = np.zeros((neq, ndis))
    # Bucle de gradientes
    for i in range(ndis):
        idx = inddis[i]
        if exactgradient and gradcomponents[idx] == 1:
            wt[i], stats = gradt(switchfun, X, Y, idx, min(1.e-8, tol / 10), stats)
            w[:, i] = gradswitchfun(X, Y, idx)
            stats[10] += 1  # stats(11) en MATLAB -> índice 10 en Python
        else:
            wt[i], stats = gradt(switchfun, X, Y, idx, min(1.e-8, tol / 10), stats)
            w[:, i], stats = graddif(switchfun, X, Y, idx, min(1.e-8, tol / 10), stats)

    # Extremos del intervalo
    if xdis > X + H:
        x0 = X + H
        x1 = min(X + 1.1 * H, xend)
    else:
        x0 = X + 0.9 * H
        x1 = X + H

    # Extremo inferior
    res0 = estirapro(FUN, switchfun, inddis, X, Y, WRK, H, x0, stats, wt, w, tol, rundata)
    endslid0, gg0, stats = res0[6], np.copy(res0[7]), res0[10]
    gout = np.copy(gg0)
    
    gg0[inddis] = 1000  
    value0 = np.min(np.abs(gg0))

    # Extremo superior
    res1 = estirapro(FUN, switchfun, inddis, X, Y, WRK, H, x1, stats, wt, w, tol, rundata)
    y1, f1, xpro, ypro, disctype, indsliding, endslid1, gg1, isterminal, direction, stats = res1
    endslid1=np.atleast_1d(endslid1)
    gg1=np.atleast_1d(gg1)
    gg1[inddis] = 1000  
    gcond = (gg1 * gg0 < 0) * (gg1 * direction >= 0)
    endfilippov = np.any(endslid1[:ndis] > 0)
    newdiscon = np.any(gcond)
    
    ind = 0
    if newdiscon:
        ind = np.where(gcond == 1)[0]  #  new discon index
        
    indexit = 0
    if endfilippov:
        indexit = np.where(endslid1[:ndis] > 0)[0]  # possible exit sliding discon index

    xout = xpro
    yout = np.copy(ypro)
    
    # Determinar tipo de punto de cruce
    if disctype == -7:
        value1 = -1
        switchpoint = 3
        indexit = 0 
    elif endfilippov and newdiscon:  # exit sliding and new discon
        value1 = -min(np.min(np.abs(gg1[ind])), np.abs(np.min(endslid1[indexit])))
        switchpoint = 4
    elif newdiscon:   #  new discon
        value1 = -np.min(np.abs(gg1[ind]))
        switchpoint = 1
    elif endfilippov:  # exit sliding
        value1 = -np.abs(np.min(endslid1[indexit]))
        switchpoint = 2
    else:                     # no discon at all
        if Verbose >= 0:
            print(f"\n Phantom pro X {X}")
            print(f"\n Phantom pro Discon endslid0 !! {endslid0}")
            print(f"\n Phantom pro Discon endslid1 !! {endslid1}")
        xout = X
        yout = np.copy(Y)
        integration_flow = 2
        Hout = H / 4
        input("Presiona Enter para continuar... (Pause)")
        return (WRKout, xout, yout, gout, Hout, integration_flow, xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)

    valueb = value1

    # Bucle de la Secante
    xx1, xx2 = x0, x1
    xnew, ynew = x1, y1
    endslid = np.copy(endslid1)
    
    
    eps_x1 = np.spacing(x1)
    eps_1 = np.spacing(1.0)
    
    if abs(x1 - x0) < 10 * max(eps_x1, eps_1):
        xnew = x1
        if Verbose >= 0:
            print(f"\n Warning very small step size in Finddiscpro !!!! {X} {x0} {x1} {Y[0]}")
    elif abs(value1) < 1.e-18:
        xnew = x1

    ii = 1
    value = value1
    if Verbose >= 2:
        print('\n Refining discontinuity with secant method in finddiscpro')
    while (abs(xx2 - xx1) >= 100 * np.spacing(max(abs(x1), 1.0)) or (switchpoint == 1 and valueb <= -1.e-6)) and ii < 160 and valueb <= -1.e-14:
        if abs(value1 - value0) > 1.e-15 and ii <= 30:
            xnew = x1 - value1 * (x1 - x0) / (value1 - value0)
        else:
            xnew = (xx1 + xx2) / 2.0
            
        if xnew >= xx2 or xnew <= xx1:
            xnew = (xx1 + xx2) / 2.0
        res_loop = estirapro(FUN, switchfun, inddis, X, Y, WRK, H, xnew, stats, wt, w, tol, rundata)
        ynew, fnew, xplus, yplus, disctype, indsliding1, endslid1, gnew, isterminal, direction, stats = res_loop
        endslid1=np.atleast_1d(endslid1)
        gnew=np.atleast_1d(gnew)
        gaux = np.copy(gnew)
        gnew[inddis] = 1000
        gcondp = (gnew * gg0 <= 0) * (gnew * direction >= 0)
        gcondp[inddis] = 0
    
        
        endfilippov = np.any(endslid1[:ndis] > 0)
        if endfilippov:
            indexit = np.where(endslid1[:ndis] > 0)[0]
            
        newdiscon = np.any(gcondp)
        if newdiscon:
            ind = np.where(gcondp == 1)[0]

        if disctype == -7:
            value = -1
            switchpoint = 3
        elif endfilippov and newdiscon:
            value = -min(np.min(np.abs(gnew[ind])), np.abs(np.min(endslid1[indexit])))
            switchpoint = 4
        elif newdiscon:
            value = -np.max(np.abs(gnew[ind]))
            switchpoint = 1
        elif endfilippov:
            value = -np.abs(np.min(endslid1[indexit]))
            switchpoint = 2
        elif switchpoint == 1:
            value = np.min(np.abs(gnew[ind]))
        elif switchpoint == 4:
            value = min(np.min(np.abs(gnew[ind])), abs(endslid1[0]))
        elif switchpoint == 2:
            value = np.abs(np.min(endslid1[indexit])) + 2 * np.spacing(1.0)
        elif switchpoint == 3:
            value = np.abs(np.min(endslid1[indexit]))
        else:
            value = np.min(np.abs(gnew[ind]))

        # Actualizar intervalo
        if value <= 0:
            xx2 = xnew
            f1 = fnew
            xout = xplus
            yout = np.copy(yplus)
            indsliding = np.copy(indsliding1)
            endslid = np.copy(endslid1)
            valueb = value
            gout = np.copy(gaux)
            disctypeout=disctype
        else:
            xx1 = xnew

        x0, value0 = x1, value1
        x1, value1 = xnew, value
        ii += 1
    # Guardar trayectorias tdis e ydis
    tdis = list(tdis) + [xnew]
    ydis = np.vstack([ydis, ynew.flatten()])
    # Manejo de almacenamiento de resultados (xx, yy)
    tspan_arr = np.atleast_1d(tspan)
    if len(tspan_arr) > 2 and solyes > 1:
        tnew = np.where((tspan_arr > X) & (tspan_arr <= xnew))[0]
        for idx_t in tnew:
            nout += 1
            xtspan = tspan_arr[idx_t]
            Y2 = estirapro(FUN, switchfun, inddis, X, Y, WRK, H, xtspan, stats, wt, w, tol, rundata)[0]
            xx[idx_t] = xtspan
            yy[idx_t, :] = Y2.flatten()
            print("\n nout",nout, xtspan)
    else:
        if Refine >= 2:
            for irefine in range(1, Refine):
                xrefine = X + (xnew - X) * irefine / Refine
                Y2 = estirapro(FUN, switchfun, inddis, X, Y, WRK, H, xrefine, stats, wt, w, tol, rundata)[0]
                nout += 1
                xx[nout-1] = xrefine  
                yy[nout-1, :] = Y2.flatten()
                print("\n nout",nout, xrefine)
        nout += 1
        xx[nout-1] = xnew
        yy[nout-1, :] = ynew.flatten()
        print("\n nout",nout, xnew)

    if ii >= 160 and Verbose >= 0:
        print(f"\n Warning !!! Too many secant iterations, {ii} {value} {xx2-xx1}")

    if Verbose >= 2:
        print(f"\n Secant finished {xnew}")
    if Verbose >= 1:
        print(f"\n discontinuity found (sliding) at X={xnew}, disctype={disctype}, switchpoint={switchpoint}")
    # Análisis del camino a proceder final
    indsliding=np.atleast_1d(indsliding)
    disctype=disctypeout
    if (switchpoint == 1 or switchpoint == 4) and len(indsliding) > len(inddis) and isterminal[indsliding[-1]] < 0:
        yout = doatswitch(xnew, ynew)
        xout = xnew
        integration_flow = 5
        return (WRKout, xout, yout, gout, Hout, integration_flow, xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)
        
    elif (switchpoint == 1 or switchpoint == 4) and len(indsliding) > len(inddis) and isterminal[indsliding[-1]] == 1:
        integration_flow = -2
        idis = list(idis) + list(inddis+1)
        stats[5] += 1
        if Verbose >= 1:
            print('\n the switching point ends the integration')
        xout, yout = xnew, ynew
        return (WRKout, xout, yout, gout, Hout, integration_flow, xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)
        
    elif switchpoint == 3:
        integration_flow = -5
        idis = list(idis) + list(inddis+1)
        stats[5] += 1
        if Verbose >= 1:
            print('\n the tangent switching point ends the integration')
        xout, yout = xnew, ynew
        return (WRKout, xout, yout, gout, Hout, integration_flow, xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)
    if disctype == -5:
        integration_flow = 4
        return (WRKout, xout, yout, gout, Hout, integration_flow, xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)
    elif disctype == -7:
        integration_flow = 7
        return (WRKout, xout, yout, gout, Hout, integration_flow, xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)

    if switchpoint == 1:
        integration_flow = 2
        WRKout[:, 0] = f1.flatten()
        if len(indsliding) == len(inddis):
            indsliding = np.copy(inddis)
            idis = list(idis) + list(ind+1)
            if Verbose >= 2:
                print(f'\n Transversal discontinuity inside sliding region at X= {xnew}')
            return (WRKout, xout, yout, gout, Hout, integration_flow, xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)
        else:
            idis = list(idis) + list(-(ind+1))
            stats[6] += 1
            if Verbose >= 3:
                print(f'\n New sliding point at X= {xnew}')
            return (WRKout, xout, yout, gout, Hout, integration_flow, xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)

    elif switchpoint == 2:
        WRKout[:, 0] = f1.flatten()
        if np.all(endslid >= 0):
            integration_flow = 0
            idis = list(idis) + list(-(inddis+1))
            stats[7] += 1
            if Verbose >= 3:
                print(f'\n Exit from all sliding at X= {xnew}')
            indsliding = np.array([])
        else:
            integration_flow = 2
            idis = list(idis) + list(inddis+1)
            stats[7] += 1
            if Verbose >= 3:
                print(f'\n Exit from some manifold sliding at X= {xnew}')
                
        idx_t = indexit[0] if hasattr(indexit, '__len__') else indexit
        paso = np.sign(gout[inddis[idx_t]]) * max(2 * np.spacing(X), min(tol / 10, 1.e-8)) / max(1.0, abs(wt[idx_t]), np.linalg.norm(w[:, idx_t]))
        xout = xout + paso * wt[idx_t]
        yout = yout + paso * w[:, idx_t]
        
        if len(indsliding) > len(indexit):
            wt_list = list(wt)
            w_list = [w[:, k] for k in range(w.shape[1])]
            for i_rm in sorted(indexit, reverse=True):
                del wt_list[i_rm]
                del w_list[i_rm]
            wt = np.array(wt_list)
            w = np.array(w_list).T if w_list else np.empty((neq, 0))
            ll = np.zeros(ndis - len(indexit))
            res_proj = proj(FUN, switchfun, indsliding, xout, yout, wt, w, ll, stats, tol, rundata)
            ypro, l1, fout, xout, yout, disctype, indsliding1, endslid, gout, isterminal, direction, errorproj, stats = res_proj
 #       else:
 #           fout = FUN(xout, yout)
        return (WRKout, xout, yout, gout, Hout, integration_flow, xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)

    elif switchpoint == 4:
        if np.all(endslid > 0):
            WRKout[:, 0] = f1.flatten()
            integration_flow = 0
            idis = list(idis) + list(-(inddis+1))
            stats[7] += 1
            if Verbose >= 3:
                print(f'\n Exit from all sliding and new transversal discontinuity at X= {xnew}')
            return (WRKout, xout, yout, gout, Hout, integration_flow, xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)
        else:
            WRKout[:, 0] = f1.flatten()
            integration_flow = 2
            idis = list(idis) + list(-(inddis+1))
            stats[7] += 1
            if Verbose >= 3:
                print(f'\n Exit from some sliding and new discontinuity at X= {xnew}')
            return (WRKout, xout, yout, gout, Hout, integration_flow, xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)

    print(f'\n Caso no contemplado en finddiscpro {switchpoint}')
    integration_flow = -5
    return (WRKout, xout, yout, gout, Hout, integration_flow, xx, yy, tdis, ydis, idis, indsliding, nout, npoints, stats)
#
# End of Finddiscpro
#
def proj(FUN, switchfun, inddis, x, y, wt, w, ll, stats, tolrk, rundata):
    """
    Traducción a Python de la función 'proj'.
    Proyecta un punto (x, y) sobre la superficie de conmutación y luego lo clasifica.
    """
    # Extraer variables necesarias de rundata
    Verbose = rundata.Verbose

    # Asegurar que las entradas sean arreglos de NumPy si es necesario
    y = np.asarray(y, dtype=float)
    wt = np.asarray(wt, dtype=float)
    w = np.asarray(w, dtype=float)
  #  ll = np.asarray(ll, dtype=float)
    

    # 1. Llamar a la función de proyección 'projnew' (función externa)

    xpro, ypro, gout, isterminal, direction, errorproj, l1, stats = projnew(
        switchfun, inddis, x, y, wt, w, ll, stats, Verbose
    )
    # Si hay un error en la proyección, abortar y devolver valores por defecto
    if errorproj > 0:
        ff = 0.0
        xout = x
        yout = x  # En MATLAB está asignado como 'yout = x'
        disctype = -10
        indsliding = 0
        endslid = -10.0
        return (ypro, l1, ff, xout, yout, disctype, indsliding, 
                endslid, gout, isterminal, direction, errorproj, stats)

    # 2. Si la proyección fue exitosa, clasificar el punto con 'classifypoint' (función externa)
    xout, yout, ff, gout, disctype, indsliding, endslid, stats = classifypoint(
        xpro, ypro, inddis, FUN, switchfun, stats, tolrk, rundata
    )
    return (ypro, l1, ff, xout, yout, disctype, indsliding, 
            endslid, gout, isterminal, direction, errorproj, stats)
#
#  End of proj
#

def rkpronew(FUN, switchfun, idis, X, Y, gxy, WRK, H, stats, wt, w, tol, endslid0, rundata):
    """
    Traducción a Python de la función 'rkpronew' (DOPRI 5(4) proyectado).
    Asegura indexación base 0, compatibilidad de dimensiones con NumPy y preserva el flujo lógico.
    """
    # Extraer parámetros desde 'rundata'
    minfortangent = np.array([rundata.minfortangent1, rundata.minfortangent2])
    Verbose = rundata.Verbose
    EABS = rundata.AbsTol
    EREL = rundata.RelTol
    XEND = rundata.Xend

    # Asegurar que los datos sean arrays de NumPy con el tipo correcto
    Y = np.asarray(Y, dtype=float)
    gxy = np.asarray(gxy, dtype=float).copy()
    WRK = np.asarray(WRK, dtype=float)
    idis = np.asarray(idis, dtype=int)
    # Convertir índices de idis de base 1 (MATLAB) a base 0 (Python)
    idis_0 = idis

    # Matriz de coeficientes Runge-Kutta Butcher (A) - 7x7
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0.108029, 0, 0, 0, 0, 0, 0],
        [0.0405108750, 0.1215326250, 0, 0, 0, 0, 0],
        [0.0607663125, 0.0, 0.1822989375, 0, 0, 0, 0],
        [0.297105016920909930065219608547, 0.0, -0.969346938437952801300701607363, 1.18312292151704287123548199882, 0, 0, 0],
        [-0.445767719936558689939662812961, 0, 2.74214758450584659845562912716, -2.34407023022786831138086780967, 0.801260365658580402864901495470, 0, 0],
        [0.166870931891601717285119862416, 0, -0.735371113362101801736678191285, 1.20597299548071395090925872364, -0.136798, 0.399325185989786133542299605231, 0]
    ])
    
    # Vectores de pesos
    C = np.array([0.0, 0.108029, 324087.0/2000000.0, 972261.0/4000000.0, 0.510881, 0.75357, 0.9])
    B = np.array([0.0835287062817292866236614389718, 0.0, 0.0, 0.306545998706544449984758902241, 
                  0.267965179973423262607667345489, 0.130936983245107245109927623181, 0.211023131793195755673984690118])
    B1 = np.array([-0.000882201938392711047700974918454, 0, 0, 0.632177414748733110758337278349, 
                   -0.282232701759180326509200083357, 0.650937488948839926798563779927, 0])

    XPH = X + H
    WRKout = WRK.copy()
    gout = gxy.copy()
    xdis = X + H
    ndis = len(idis)
    ll = np.zeros(ndis)
    
    # stats(10) en MATLAB -> stats[9] en Python
    stats[9] += 1
    gxy[idis_0] = 1000.0
    
    stagedis = 8
    
    # MATLAB: K=2:7 -> Python: range(1, 7) que equivale a K=1..6 (en base 0)
    for K in range(1, 7):
        HH = H * C[K]
        # Multiplicación matricial rápida utilizando @
        Y1 = Y + H * (WRK[:, :K] @ A[K, :K])
        
        # Llamada a proj (función externa)
        _, l1, ff, _, _, disctype, _, endslid1, gyx, _, direction, errorproj, stats = proj(
            FUN, switchfun, idis, X + HH, Y1, wt, w, ll, stats, tol, rundata
        )
        
        if errorproj > 0:
            ERR = 3.0
            return XPH, Y1, gout, WRKout, ERR, disctype, xdis, stagedis, stats
            
        if np.any(endslid1 > 10**5):
            ERR = 2.0
            return XPH, Y1, gout, WRKout, ERR, disctype, xdis, stagedis, stats
        
            
        ll = l1.copy()
        WRK[:, K] = ff.T
        
  #      gyx = np.asarray(gyx, dtype=float).copy()
        gyx=np.atleast_1d(gyx)
        gyx[idis_0] = 1000.0
        
        auxa = (np.sign(gyx) * np.sign(gxy)) < 0
        auxb = (gyx * direction) >= 0
        
        cond_endslid = np.any(endslid1[:ndis] > minfortangent[0])
        cond_cross = np.any(auxa * auxb)
        
        if cond_endslid or cond_cross or disctype == -7:
            condtrans = cond_cross
            if condtrans:  # Una nueva discontinuidad ha sido cruzada
                if Verbose >= 3:
                    iaux = np.where((auxa * auxb) == True)[0]
                    # Se suma 1 a iaux para coincidir con la visualización en base 1 de MATLAB
                    print(f"\n New discontinuity? codtrans, K, iaux , g {int(condtrans)} {K+1} {iaux+1} {gxy[iaux]} {gyx[iaux]}")
                disctype = 5
            else:  # Posible fin de deslizamiento encontrado
                if Verbose >= 3:
                    indexit = np.where((endslid1[:ndis] > minfortangent[0]) == True)[0]
                    print(f"\n end of sliding? condtrans, stage, indexit, endslid {int(condtrans)} {K+1} {indexit+1} {endslid1}")
                disctype = 4
                
            ERR = 3.0
            stagedis = K + 1  # Retornar en base 1 para consistencia con MATLAB
            xdis = X + H * C[K]
            return XPH, Y1, gout, WRKout, ERR, disctype, xdis, stagedis, stats

    # Estimación de error
    EST = 0.0
    Y1 = Y.copy()
    for K in range(7):
        EST = EST + H * WRK[:, K] * (B[K] - B1[K])
        
    ERR = np.max(np.abs(EST))
    TOL = EABS + EREL * np.max(np.abs(Y))
    
    if ERR > TOL:
        XPH = X + H
        disctype = 3
        return XPH, Y1, gout, WRKout, ERR, disctype, xdis, stagedis, stats
    # Llamada a estirapro (función externa)
    Y1, ff, _, _, disctype, _, endslid1, gyx, _, direction, stats = estirapro(
        FUN, switchfun, idis, X, Y, WRK, H, X + H, stats, wt, w, TOL, rundata
    )
    
    if disctype == -10:
        ERR = 3.0
        return XPH, Y1, gout, WRKout, ERR, disctype, xdis, stagedis, stats

 #   gyx = np.asarray(gyx, dtype=float).copy()
    gyx=np.atleast_1d(gyx)
    gyx[idis_0] = 1000.0
    
    auxa = (np.sign(gyx) * np.sign(gxy)) < 0
    auxb = (gyx * direction) >= 0
    condtrans = np.any(auxa * auxb)
    
    if np.any(endslid1 > 10**5):
        ERR = 2.0
        return XPH, Y1, gout, WRKout, ERR, disctype, xdis, stagedis, stats
        
    if np.any(endslid1[:ndis] > minfortangent[0]) or condtrans or disctype == -7:
        xdis = X + H
        disctype = 5
        WRKout = WRK.copy()
        if Verbose >= 3:
            print(f"\n Projecting,  endslid0, endslid1, {endslid0} {endslid1}")
            print(f"\n Projecting, condtrans, H, {int(condtrans)} {H}")
        return XPH, Y1, gout, WRKout, ERR, disctype, xdis, stagedis, stats

    XPH = X + H
    # WRK(:, 8) en MATLAB es WRK[:, 7] en Python
    WRK[:, 7] = ff
    WRKout = WRK.copy()

    if ERR > (TOL / 10.0) or (X + 1.1 * H) > XEND:
        disctype = 3
        return XPH, Y1, gout, WRKout, ERR, disctype, xdis, stagedis, stats

    # Segunda llamada a estirapro
    _, _, _, _, disctype, _, endslid1, gyx, _, direction, stats = estirapro(
        FUN, switchfun, idis, X, Y, WRK, H, X + 1.1 * H, stats, wt, w, TOL, rundata
    )
    
    if disctype == -10 or disctype == -7:
        ERR = 3.0
        return XPH, Y1, gout, WRKout, ERR, disctype, xdis, stagedis, stats
        
    gyx = np.atleast_1d(gyx)
    gyx[idis_0] = 1000.0
    
    auxa = (np.sign(gyx) * np.sign(gxy)) < 0
    auxb = (gyx * direction) >= 0
    condtrans = np.any(auxa * auxb)
    
    if np.any(endslid1 > 10**5):
        ERR = 2.0
        return XPH, Y1, gout, WRKout, ERR, disctype, xdis, stagedis, stats
        
    if np.any(endslid1[:ndis] > minfortangent[0]) or condtrans:
        xdis = X + 1.1 * H
        disctype = 5
        if Verbose >= 3:
            print(f"\n Projecting plus,  endslid0, endslid1, {endslid0} {endslid1}")
            print(f"\n Projecting plus, condtrans, H, {int(condtrans)} {H}")
        return XPH, Y1, gout, WRKout, ERR, disctype, xdis, stagedis, stats

    return XPH, Y1, gout, WRKout, ERR, disctype, xdis, stagedis, stats
#
#  End of RKPONEW
#

def slide(FUN, switchfun, H, X, Y, WRK, g0, xx, yy, indsliding, endslid, 
          nout, npoints, stats, rundata):
    """
    Traducción a Python de la función 'slide' de MATLAB.
    Preserva el flujo de control, la indexación corregida (0-based) y las operaciones numéricas.
    """
    
    # Extraer campos de la estructura de datos 'rundata' (asumiendo que es un objeto o diccionario)
    # Si 'rundata' es un diccionario, usa rundata['Verbose'], etc. Aquí se asume formato de objeto/clase.
    Verbose = rundata.Verbose
    gradcomponents = rundata.gradcomponents
    XEND = rundata.Xend
    tspan = np.asarray(rundata.tspan)
    exactgradient = rundata.exactgradient
    EABS = rundata.AbsTol
    EREL = rundata.RelTol
    gradswitchfun = rundata.Gradient
    Refine = rundata.Refine
    solyes = rundata.nargout

    # Asegurar que Y, WRK, xx, yy sean arrays de NumPy
    Y = np.asarray(Y, dtype=float)
    WRK = np.asarray(WRK, dtype=float)
    xx = np.asarray(xx, dtype=float)
    yy = np.asarray(yy, dtype=float)
    indsliding = np.atleast_1d(indsliding)

    REJECT = False
    advance = True
    xout = X
    yout = Y.copy()
    Hout = H
    gout = g0
    integration_flow = -5
    WRKout = WRK.copy()
    inddis = np.atleast_1d(indsliding)
    irestart = 0
    ndis = len(inddis)
    
    ll = np.zeros(ndis)
    wt = np.zeros(ndis)
    w = np.zeros((Y.shape[0], ndis))
    tol = EABS + EREL * np.max(np.abs(Y))
    xdis = X
    if X >= XEND:
        integration_flow = -5
        xdis = X
        return WRKout, xout, yout, gout, Hout, integration_flow, xdis, xx, yy, nout, npoints, stats

    neq = Y.shape[0]
    chunk = int(min(max(100, 50 * Refine), Refine + np.floor((2**13) / neq))+100)

    while advance and X < XEND:
        H = min(H, XEND - X)
        TOL = EABS + EREL * np.max(np.abs(Y))
        
        # En Python, 'eps' equivalente a eps de MATLAB se obtiene con np.finfo(float).eps
        eps_X = np.finfo(float).eps * abs(X) if X != 0 else np.finfo(float).tiny
        eps_val = np.finfo(float).eps * 0.001
        
        if H < max(5 * eps_X, eps_val):
            if Verbose >= 0:
                print(f"\n Minimum step size h={H} when sliding attained at X= {X} \n Integration stopped \n")
            xdis = X
            integration_flow = -1
            return WRKout, xout, yout, gout, Hout, integration_flow, xdis, xx, yy, nout, npoints, stats

        if not REJECT:  # It is not a rejected step
            for i in range(ndis):
                # Restamos 1 a indsliding[i] para la indexación base 0 si es necesario. 
                # (Asumimos que indsliding viene de MATLAB indexado desde 1, ajustamos si es necesario)
                idx_sliding_i = indsliding[i] - 1 
                
                if exactgradient and gradcomponents[idx_sliding_i] == 1:
                    # Llamada a gradt (función externa)
                    wt[i], stats = gradt(switchfun, X, Y, indsliding[i], min(1e-8, tol/10), stats)
                    # Llamada directa a gradswitchfun (reemplaza feval de MATLAB)
                    w[:, i] = gradswitchfun(X, Y, indsliding[i])
                    stats[10] += 1  # stats(11) en MATLAB es stats[10] en Python
                else:
                    wt[i], stats = gradt(switchfun, X, Y, indsliding[i], min(1e-8, tol/10), stats)
                    w[:, i], stats = graddif(switchfun, X, Y, indsliding[i], min(1e-8, tol/10), stats)

        # Integrador RK (función externa)
        XPH, Y1, gout, WRKout, ERR, disctype, xdis, stagedis, stats = rkpronew(
            FUN, switchfun, inddis, X, Y, g0, WRK, H, stats, wt, w, TOL, endslid, rundata
        )
        WRK = WRKout.copy()
        
      

        if Verbose >= 4:
            print(f"\n Y {Y.flatten()} {disctype}")
            print(f"\n primera etapa {WRK[:, 0]}")
            print(f"\n segunda etapa {WRK[:, 1]}")
            print(f"\n tercera etapa {WRK[:, 2]}")
            print(f"\n cuarta etapa {WRK[:, 1]}")

        if disctype == 3 or XPH == X:  # There is not an exit point or it is a first step
            TOL = EABS + EREL * np.max(np.abs(Y))
            if ERR <= TOL:
                # Accepted sliding step
                if Verbose >= 2:
                    print(f"\n aceptado slide X1, Y1,H, ERR, TOL {XPH:8.6g} {np.linalg.norm(Y1)} {H:e} {ERR:e} {TOL:e}")

                FAC = min(0.9 * (TOL / (ERR + 1e-16))**(1./5.), 2.4)
                if REJECT:
                    FAC = min(FAC, 1.0)

                if len(tspan) > 2 and solyes > 1:
                    tnew = np.where((tspan > X) & (tspan <= XPH))[0]
                    for ii in range(len(tnew)):
                        xtspan = tspan[tnew[ii]]
                        Y2, f1, xplus, yplus, disctype,indsliding, endslid, gxy, isterminal, direction, stats= estirapro(
                            FUN, switchfun, inddis, X, Y, WRK, H, xtspan, stats, wt, w, TOL, rundata
                            )
                        indsliding=np.atleast_1d(indsliding)
                        xx[tnew[ii]] = xtspan
                        yy[tnew[ii], :] = np.asarray(Y2)
                        nout += 1
                        print("\n nout",nout, xtspan)
                else:
                    if nout + Refine > npoints-1:
                        npoints += chunk
                        xx = np.append(xx, np.zeros(chunk))
                        yy = np.vstack([yy, np.zeros((chunk, neq))])

                    if Refine >= 2:
                        for irefine in range(1, Refine):
                            xrefine = X + (XPH - X) * irefine / Refine
                            Y2 = estirapro(FUN, switchfun, inddis, X, Y, WRK, H, xrefine, stats, wt, w, TOL, rundata)
                            nout += 1
                            xx[nout - 1] = xrefine
                            yy[nout - 1, :] = Y2[0]
                            print("\n nout",nout, xrefine)

                    nout += 1
                    xx[nout - 1] = XPH
                    yy[nout - 1, :] = Y1
                    print("\n nout",nout, XPH)

                # Equivalente a WRK(:,1) = WRK(:, 8) -> Python: índice 0 e índice 7
                WRK[:, 0] = WRK[:, 7]
                g0 = gout
                stats[2] += 1  # stats(3) en MATLAB -> stats[2] en Python
                X = XPH
                Y = Y1.copy()
                H = FAC * H
                REJECT = False

                # Límite con precisión de máquina
                eps_val_X = np.finfo(float).eps * max(1.0, abs(X))
                if (X - XEND) + 5.0 * eps_val_X > 0.0:
                    X = XEND
                    yout = Y.copy()
                    xout = X
                    integration_flow = -5
                    return WRKout, xout, yout, gout, Hout, integration_flow, xdis, xx, yy, nout, npoints, stats
                
                irestart = 0
            else:
                # Rejected sliding step
                if Verbose >= 2:
                    print(f"\n rechazo slide X, H, ERR, TOL {X:8.4g} {H:e} {ERR:e} {TOL:e} {disctype}")

                if ERR == 3.0:
                    FAC = 0.5
                else:
                    FAC = max(0.9 * (TOL / (ERR + 1e-12))**(1./5.), 0.10)
                
                REJECT = True
                H = FAC * H
                stats[3] += 1  # stats(4) -> stats[3]
        else:
            # Possible exit of sliding detected at this step
            stats[4] += 1  # stats(5) -> stats[4]
            if Verbose >= 2:
                print(f"\n rechazo discon slide X, H, disctype, Stage {X} {H} {disctype} {stagedis}")
            
            irestart += 1
            paso = min(max([1000 * (np.finfo(float).eps * abs(X) if X != 0 else np.finfo(float).tiny), 2.22e-16, TOL / 10000]), XEND - X)
            
            if stagedis > 7 and ERR <= TOL:
                xout = X
                yout = Y.copy()
                Hout = H
                integration_flow = 3
                return WRKout, xout, yout, gout, Hout, integration_flow, xdis, xx, yy, nout, npoints, stats
                
            elif stagedis > 7 and ERR > TOL:
                FAC = min(0.5, max(0.9 * (TOL / (ERR + 1e-12))**(1./5.), 0.10))
                H = FAC * H
                REJECT = True
                
            elif stagedis >= 10 or (irestart >= 6 and stagedis == 2) or H < max(100 * (np.finfo(float).eps * abs(X) if X != 0 else np.finfo(float).tiny), 2.22e-16) or (irestart >= 4 and X == xout and stagedis <= 2):
                print(f"\n Bad restarting or problems in slide irestart, disctype, stagedis, {irestart} {disctype} {stagedis} {X} {H}")
                
                # En lugar de "pause" de MATLAB, usamos input() para pausar la ejecución en consola
                input("Presiona Enter para continuar...") 
                
                integration_flow = 5
                npasos = 2
                ii = 0
                xout = X
                yout = Y.copy()
                error = 100.0
                
                while error > TOL or ii < npasos:
                    xout = xout + paso
                    yout = yout + paso * WRK[:, 0]
                    
                    # Llamar a la proyección
                    ypro, _, WRK_col1, xplus, yplus, disctype, indsliding1, endslid, gxy, _, _, errorproj, stats = proj(
                        FUN, switchfun, inddis, xout, yout, wt, w, ll, stats, tol, rundata
                    )
                    WRK[:, 1] = WRK_col1  # Actualizar columna en Python (WRK(:,2) es WRK[:, 1])
                    
                    error = paso * np.linalg.norm(WRK[:, 1] - WRK[:, 0]) / 2.0
                    
                    if errorproj > 0:
                        print("\n no projection convergence in slide")
                        error = 100.0
                        paso = paso / 2.0
                    elif error > TOL:
                        paso = max(paso / 10.0, 0.9 * paso * TOL / error)
                    else:
                        ii += 1
                        WRK[:, 0] = WRK[:, 1].copy()
                        paso = min([5.0 * paso, 0.95 * paso * TOL / error, XEND - X])
                        
                        if disctype == 3:
                            if len(indsliding1) > ndis:
                                print(inddis)
                                print(indsliding1)
                            
                            inddis = indsliding1.copy()
                            
                            if len(indsliding1) < ndis:
                            
                            # Réplica de ismember de MATLAB para filtrar vectores por índices en común
                                index = np.isin(indsliding, indsliding1)
                                ndis = len(indsliding1)
                                indsliding = np.asarray(indsliding1, dtype=int)
                                wt = wt[index]
                                w = w[:, index]
                                H = paso
                                g0 = gxy
                                break
                        
                        elif disctype == 1:
                            xout = xplus
                            yout = yplus.copy()
                            gout = gxy
                            WRKout = WRK.copy()
                            Hout = H
                            integration_flow = 0
                            return WRKout, xout, yout, gout, Hout, integration_flow, xdis, xx, yy, nout, npoints, stats
                
                if disctype != 3:
                    yout = ypro.copy()
                    Hout = paso
                    return WRKout, xout, yout, gout, Hout, integration_flow, xdis, xx, yy, nout, npoints, stats
                
                irestart = 0
                X = xout
                Y = yout.copy()
                H = paso
            else:
                H = xdis - X
                REJECT = True

    if XEND <= X:
        integration_flow = -1
        
    WRKout = WRK.copy()
    xout = X
    return WRKout, xout, yout, gout, Hout, integration_flow, xdis, xx, yy, nout, npoints, stats
#
# End of slide
#

def estirapro(FUN, switchfun, idis, X, Y, WRK, H, XA, stats, wt, w, tol, rundata):
    """
    This function uses the continuous extension of the step
    [X, X+H] to get an approximation of order five at the
    point XA, with XA in [X,X+H+0.3*H]
    """
    
    # En Python se usa la primera dimensión (filas) o shape[1] para columnas.
    # size(idis, 2) en MATLAB obtiene el número de columnas.
 #   ndis = idis.shape[0] if idis.ndim > 1 else 1 
    ndis=idis.size
    ll = np.zeros(ndis)
    
    t = (XA - X) / H
    
    # Inicializamos el vector de coeficientes con 7 elementos (0 a 6 en Python)
    co = np.zeros(7)
    
    co[0] = (0.0835287062817292866236614389718 +
             0.10833226382936095050591041754 * (t - 1) + 
             0.94376671763641998485566302463 * (t - 1)**2 + 
             3.27141986295009036688531952391 * (t - 1)**3 + 
             4.72722509689971814810898444919 * (t - 1)**4 + 
             2.37476839403841610219707897135 * (t - 1)**5)
             
    co[1] = 0.0
    co[2] = 0.0
    
    co[3] = (0.306545998706544449984758902241 -
             0.55211188975166266877095507565 * (t - 1) -
             4.721230159426937567153330024 * (t - 1)**2 -
             15.7210693670286862319079470108 * (t - 1)**3 - 
             21.01962580831593268736351876355 * (t - 1)**4 -
             9.16112871225597690385318779876 * (t - 1)**5)
             
    co[4] = (0.267965179973423262607667345489 +
             1.44365568758696219783727880618 * (t - 1) + 
             11.82286540877065862510085608089 * (t - 1)**2 + 
             35.8655781733858495754882454935 * (t - 1)**3 + 
             40.6973569709404557539847012416 * (t - 1)**4 + 
             15.4789536987117258683677003683 * (t - 1)**5)
             
    co[5] = (0.130936983245107245109927623181 -
             2.70812954093109281924594200438 * (t - 1) -
             19.45196651227625845774250951659 * (t - 1)**2 -
             44.8851943969347913581527622554 * (t - 1)**3 -
             42.9016923369907148456114020941 * (t - 1)**4 -
             14.6293979281559818808452797278 * (t - 1)**5)
             
    co[6] = (0.211023131793195755673984690118 +
             2.70825347926643233967370785603 * (t - 1) +
             11.40656454529611741493932043467 * (t - 1)**2 +
             21.46926572762753764768714424857 * (t - 1)**3 + 
             18.4967360774664736308812351667 * (t - 1)**4 +
             5.93680454766181681413368818688 * (t - 1)**5)
             
    # WRK(:,1:7) en MATLAB toma las primeras 7 columnas (0 a 6 en Python).
    # .dot() o @ realiza el producto matricial con el vector 'co'.
    cont = WRK[:, :7].dot(co)

    # Computación de la aproximación
    Y1 = Y + H * cont
    
    
    # Se asume que la función 'proj' ya está definida en Python y devuelve una tupla
    (ypro, _, f1, xplus, yplus, disctype, indsliding, endslid, 
     gxy, isterminal, direction, errorproj, stats) = proj(
         FUN, switchfun, idis, XA, Y1, wt, w, ll, stats, tol, rundata
     )
     
    if errorproj > 0:
        print("\n no projection convergence in estirapro")
        disctype = -10
        
    Y1 = np.atleast_1d(ypro)
    
    
    return Y1, f1, xplus, yplus, disctype, indsliding, endslid, gxy, isterminal, direction, stats
#
#  End of ESTIRAPRO

def projnew(switchfun, inddis, x, y, wt, w, ll, stats, Verbose):

    """
    Traducción a Python de la función 'projnew'.
    Realiza una iteración de Newton simplificada para proyectar el punto (x, y) 
    sobre la superficie de conmutación.
    """
    # Asegurar que las entradas sean arreglos de NumPy para operaciones vectoriales
   
    wt = np.asarray(wt, dtype=float)
    w = np.asarray(w, dtype=float)
    
    
    # Ajuste de índice de base 1 (MATLAB) a base 0 (Python)
    inddis_0 = np.asarray(inddis, dtype=int)
    
    
  #  y=yin[:,np.newaxis]
   

    l0 = ll.copy()  # Aproximación inicial para el parámetro
    error = 0.0
    xpro = x +  np.dot(wt,l0)
    ypro = y +   w @ l0 # Aproximación inicial para la solución proyectada
    er = 1.0
    ns = 0  # Número de iteraciones realizadas
  #  y = np.asarray(ypro, dtype=float)

    # Llamada a la función de superficie de conmutación
 
    gpro, isterminal, direction = switchfun(xpro, ypro)
    gpro = np.atleast_1d(gpro)
    
    # stats(10) en MATLAB -> stats[9] en Python
    stats[9] += 1

    # den = wt'*wt + w'*w. En Python usamos np.dot o multiplicaciones elementales + sumas
    # .item() asegura que obtengamos un escalar puro de Python si es un array de un solo elemento
    den = (np.dot(wt, wt) + np.dot(w.T, w))
 
    # Iteración de Newton simplificada
    # Usamos indexación inddis_0 para gpro
    if np.linalg.norm(gpro[inddis_0]) < 1e-14:
        er = 0.0
        l1 = ll.copy()
    else:
        l1 = l0 .copy() # Inicialización por si no entra al bucle

    eps_1 = np.finfo(float).eps  # Equivalente a eps(1.0) en MATLAB
    while er > 200 * eps_1 and ns < 39:
        # En MATLAB: l1 = l0 - den\gpro(inddis). Al ser 'den' un escalar, es una división simple.
        # Si inddis_0 tiene múltiples elementos, esto dividirá cada elemento.
        l1 = l0 - np.linalg.solve(den, gpro[inddis_0])
        
        era = er
        er = np.linalg.norm(l1 - l0)
        l0 = l1
        
    
        
        xpro = x + wt @ l1
        ypro = np.atleast_1d(y) + w @ l1.T
    #    ypro=ypro.reshape(-1)
        ns += 1
      
        
        # Detección de posible divergencia
        if er > 10.0 or (ns > 1 and er > 1000 * abs(era)):
            xpro = x
            ypro = y
            error = 1.0
            if Verbose >= 0:
                print(f"\n Warning!! possible divergence in Newton {x} {er} {era}")
            return xpro, ypro, gpro, isterminal, direction, error, l1, stats

        # Actualizar valores
        gpro, isterminal, direction = switchfun(xpro, ypro)
        gpro = np.atleast_1d(gpro)
        stats[9] += 1
        
    # Detección de no convergencia tras el límite de iteraciones
    if ns >= 39:
        error = 2.0
        if Verbose >= 0:
            print(f"\n no projection after 39 iter, {x} {gpro[inddis_0]} {er} {era} {100*eps_1} {w}")
        return xpro, ypro, gpro, isterminal, direction, error, l1, stats
    return xpro, ypro, gpro, isterminal, direction, error, l1, stats
#
#  End of projnew
#

def rkintegration(FUN, switchfun, H, X, Y, EABS, EREL, xx, yy, nout, npoints, XEND, WRK, stats, Verbose):
    """
    This function integrates the problem along a non sliding region
    """
    if Verbose >= 1:
        print(f"\n Enter non discon integration H, X, XEND {H}, {X} {XEND}")
        
    advance = True  # set to false when XEND is attained or a discontinuity is detected
    EJECUTAR = True
    REJECT = False
    UROUND = np.finfo(float).eps  # equivalente a eps(1.0) en MATLAB
    xout = X
    yout = np.copy(Y)
    neq = Y.shape[0]
    
#    ff = np.copy(WRK[:, 0])
    
    Hout = H
    TOL = EABS + EREL * np.max(np.abs(Y))
    H = TOL
    yyy = np.copy(yy.T)  # yy' transpuesto
    chunk = int(np.floor((2**13) / neq)+100)
    
    while advance:
        if EJECUTAR:  # Executed only at accepted steps
            URO1 = 5.0 * UROUND * max(1.0, abs(X))
            if ((X - XEND) + URO1) > 0.0:
                X = XEND
                yout = np.copy(Y)
                xout = X
#                integration_flow = -5
                ffout = np.copy(WRK[:, 0])
                return ffout, xx, yy, xout, yout, Hout, nout, npoints, stats
            
            if (X + H - XEND) > 0.0:
                H = XEND - X
                
        if H < max(5 * np.finfo(float).eps, 0.001 * np.finfo(float).eps): # nota: eps(0.001) en matlab aprox
            if Verbose >= 0:
                print(f"\n Minimum step size  h={H} attained at X= {X} \n  Integration stopped \n")
            yout = np.copy(Y)
            ffout = np.copy(WRK[:, 0])
 #           integration_flow = -1
            return ffout, xx, yy, xout, yout, Hout, nout, npoints, stats

        XPH, Y1, WRKout, ERR, stats = CMR54D(FUN, X, H, Y, WRK, stats)
        TOL= EABS + EREL* max(abs(Y1));
        if ERR <= TOL:
            # Accepted step
            if nout > npoints-1:
                     npoints += chunk
                     xx = np.append(xx, np.zeros(chunk))
                     yy = np.vstack([yy, np.zeros((chunk, neq))])
            if Verbose >= 2:
                print(f"\n Paso aceptado H, X, XPH {H:20.15g}   [{X:20.15g}, {XPH:20.15g}]")
                
            stats[0] = stats[0] + 1  # Accepted steps counter Naccpt (stats(1) en MATLAB)
            
   
            # WRK = WRKout  (En el original se asignaba WRKout, asumimos que WRK se actualiza o mantiene)
            ffout = np.copy(WRK[:, 0])
            Y = np.copy(Y1)
            X = XPH
            nout +=1
            xx[nout - 1] = XPH
            yy[nout - 1, :] = Y1
            print("\n nout",nout, XPH)
 #           xx = np.append(xx, X)
 #           yyy = np.column_stack((yyy, Y)) # Asegurando dimensiones correctas
            
            FAC = min(0.9 * (TOL / (ERR + 1e-17))**(1.0 / 2.0), 2.0)
            if REJECT:
                FAC = min(FAC, 1.0)
                
            H = FAC * H
            REJECT = False
            EJECUTAR = True
            
        else:
            # Rejected step
            if Verbose >= 2:
                print(f"\nPASO rechazado {H}   [{X:30.20e}, {XPH}]")
                
            FAC = max(0.9 * (TOL / (ERR + 1e-12))**(1.0 / 5.0), 0.1)
            REJECT = True
            H = FAC * H
            stats[1] = stats[1] + 1  # Normal rejected steps counter Nrejct (stats(2) en MATLAB)
            EJECUTAR = False
            
    yy = np.copy(yyy.T)
    return ffout, xx, yy, xout, yout, Hout, nout, npoints, stats

def CMR54D(FUN, X, H, Y, WRK, stats):
    """
    Advance of one step by the CMR5(4)D pair
    """
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0.108029, 0, 0, 0, 0, 0, 0],
        [0.0405108750, 0.1215326250, 0, 0, 0, 0, 0],
        [0.0607663125, 0.0, 0.1822989375, 0, 0, 0, 0],
        [0.297105016920909930065219608547, 0.0, -0.969346938437952801300701607363, 1.18312292151704287123548199882, 0, 0, 0],
        [-0.445767719936558689939662812961, 0, 2.74214758450584659845562912716, -2.34407023022786831138086780967, 0.801260365658580402864901495470, 0, 0],
        [0.166870931891601717285119862416, 0, -0.735371113362101801736678191285, 1.20597299548071395090925872364, -0.136798000000000000000000000000, 0.399325185989786133542299605231, 0]
    ])

    C = np.array([0.0, 0.108029, 324087.0/2000000.0, 972261.0/4000000.0, 0.510881, 0.75357, 0.9])
    B = np.array([0.0835287062817292866236614389718, 0.0, 0.0, 0.306545998706544449984758902241, 
                  0.267965179973423262607667345489, 0.130936983245107245109927623181, 
                  0.211023131793195755673984690118])
    B1 = np.array([-0.000882201938392711047700974918454, 0, 0, 0.632177414748733110758337278349, 
                   -0.282232701759180326509200083357, 0.650937488948839926798563779927, 0])

    for K in range(1, 7):
        HH = H * C[K]
        # Nota: en Python WRK[:, 0:K] toma las columnas de 0 hasta K-1
        Y1 = Y + np.dot(H * WRK[:, 0:K], A[K, 0:K])
        WRK[:, K] = FUN(X + HH, Y1)

    stats[8] = stats[8] + 6  # stats(9) en MATLAB es stats[8] en Python
    EST = 0.0
    Y1 = np.copy(Y)

    for K in range(7):
        EST = EST + H * np.dot(WRK[:, K], (B[K] - B1[K]))
        Y1 = Y1 + (H * np.dot(WRK[:, K], B[K]))

    WRKout = np.copy(WRK)
    ERR = np.max(np.abs(EST))
    XPH = X + H

    return XPH, Y1, WRKout, ERR, stats

def disodeset(*args):
    """
    Crea o modifica una estructura de opciones equivalente a disodeset de MATLAB.
    
    PROPIEDADES:
    - AbsTol: Tolerancia de error absoluto [escalar positivo o vector, por defecto None]
    - RelTol: Tolerancia de error relativo [escalar positivo, por defecto None]
    - Gradient: Manejador de función para el gradiente de la función de eventos
    - GradientComponents: Componentes del gradiente
    - InitialStep: Tamaño de paso inicial sugerido [escalar positivo]
    - EventControl: Tipo de control para la detección de discontinuidades
    - ActionSwitch: Función de salida instalable [function_handle]
    - Refine: Factor de refinamiento de salida [entero positivo]
    - Verbose: Tipo de información impresa durante la integración
    """
    
    # Definir valores por defecto (None o los valores que prefieras)

    options = SimpleNamespace()
    options.AbsTol=1.e-4
    options.RelTol=1.e-4
    options.Gradient = None
    options.GradientComponents=np.array([])
    options.ActionSwitch=None
    options.InitialStep = 0
    options.EventControl=0
    options.Refine=0
    options.Verbose=0
    options.nargout = 2
    
    if len(args) % 2 != 0:
        raise ValueError("Los argumentos deben venir en parejas de 'nombre', valor.")
    
    # 3. Recorrer la lista de 2 en 2 (nombre en las posiciones pares, valor en las impares)
    for i in range(0, len(args), 2):
        key = args[i]
        value = args[i+1]
        
        if hasattr(options, key):
            setattr(options, key, value) 
        else:
            raise ValueError(f"Propiedad no reconocida: '{key}'")

    return options
