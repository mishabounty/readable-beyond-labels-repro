"""Логистическая регрессия с L2 до сходимости методом Ньютона, батчем по концептам (на GPU).

Цель для каждого концепта j -- та же, что у sklearn LogisticRegression(C):
    sum_i [softplus(s_ij) - y_ij s_ij] + ||w_j||^2 / (2C),   s = X w + b, свободный член не штрафуется.
Гессиан X^T diag(p(1-p)) X + I/C накапливается в float32, система решается в float64 (Холецкий); шаг Ньютона с
поиском по Армихо отдельно для каждого концепта. Остановка: max |градиент| / n < tol. Путь по возрастающим C
с тёплым стартом. Если примеров меньше, чем признаков (пробы по 200 меткам), решается двойственная задача:
w = X^T a (теорема о представителе), система Ньютона (n+1) x (n+1) для (a, b); цель та же.
"""

import torch
import torch.nn.functional as F


def _objective(X, Y, W, lam, reg):
    S = X @ W
    return (F.softplus(S) - Y * S).sum(0) + 0.5 * lam * (reg[:, None] * W ** 2).sum(0)


def fit_dual(X, Y, C, tol=1e-6, max_iter=100):
    """То же для n < d: X -- n x (d+1) с единичным последним столбцом; возвращает W в прямой форме."""
    Xd = X[:, :-1].double()
    Yd = Y.double()
    n, m = Yd.shape
    K = Xd @ Xd.t()
    lam = 1.0 / C
    A = torch.zeros(n, m, dtype=torch.float64, device=X.device)
    b = torch.zeros(m, dtype=torch.float64, device=X.device)

    def obj(A, b):
        S = K @ A + b
        return (F.softplus(S) - Yd * S).sum(0) + 0.5 * lam * (A * (K @ A)).sum(0)

    ok = False
    for _ in range(max_iter):
        S = K @ A + b
        P = torch.sigmoid(S)
        R = P - Yd
        G = Xd.t() @ (R + lam * A)  # градиент по w в прямой форме
        if (G.abs().amax(0) / n).max() < tol and (R.sum(0).abs() / n).max() < tol:
            ok = True
            break
        D = (P * (1 - P)).t()  # m x n
        M = torch.zeros(m, n + 1, n + 1, dtype=torch.float64, device=X.device)
        M[:, :n, :n] = D[:, :, None] * K[None] + lam * torch.eye(n, dtype=torch.float64, device=X.device)
        M[:, :n, n] = D
        M[:, n, :n] = (D[:, :, None] * K[None]).sum(1)
        M[:, n, n] = D.sum(1)
        rhs = torch.cat([(R + lam * A).t(), R.sum(0)[:, None]], 1)
        sol = torch.linalg.solve(M, rhs)
        dA, db = sol[:, :n].t(), sol[:, n]
        f0 = obj(A, b)
        step = torch.ones(m, dtype=torch.float64, device=X.device)
        for _ in range(30):
            bad = obj(A - step * dA, b - step * db) > f0 + 1e-12 * f0.abs()
            if not bad.any():
                break
            step = torch.where(bad, step / 2, step)
        A, b = A - step * dA, b - step * db
    W = torch.cat([Xd.t() @ A, b[None]], 0).float()
    return W, ok


def fit(X, Y, C, W0=None, tol=1e-6, max_iter=100, chunk=24):
    """X: n x (d+1) (последний столбец -- единицы), Y: n x m из {0,1}; возвращает (W: (d+1) x m, сошлось ли)."""
    n, d1 = X.shape
    if n < d1:
        return fit_dual(X, Y, C, tol=tol)
    m = Y.shape[1]
    W = torch.zeros(d1, m, device=X.device) if W0 is None else W0.clone()
    reg = torch.ones(d1, device=X.device)
    reg[-1] = 0.0
    lam = 1.0 / C
    eye = torch.eye(d1, device=X.device, dtype=torch.float64)
    done = torch.zeros(m, dtype=torch.bool, device=X.device)
    for _ in range(max_iter):
        S = X @ W
        P = torch.sigmoid(S)
        G = X.t() @ (P - Y) + lam * reg[:, None] * W
        done |= G.abs().amax(0) / n <= tol
        active = ~done
        if not active.any():
            return W, True
        idx = active.nonzero().squeeze(1)
        delta = torch.zeros_like(W)
        for c in torch.split(idx, chunk):
            Wt = (P[:, c] * (1 - P[:, c])).t()  # k x n
            H = torch.stack([(X * w[:, None]).t() @ X for w in Wt]).double()
            H += lam * torch.diag(reg.double()) + 1e-10 * eye
            L, info = torch.linalg.cholesky_ex(H)
            sol = torch.cholesky_solve(G[:, c].t().double().unsqueeze(-1), L).squeeze(-1)
            bad = info != 0  # на всякий случай: псевдообратная
            if bad.any():
                sol[bad] = (torch.linalg.pinv(H[bad]) @ G[:, c].t().double()[bad].unsqueeze(-1)).squeeze(-1)
            delta[:, c] = sol.t().float()
        dec = (G * delta).sum(0)
        # декремент Ньютона: ниже точности float32 цели шаг уже ничего не меняет -- считаем, что сошлось
        small = active & (dec / 2 <= 1e-9 * n)
        done |= small
        active = active & ~small
        if not active.any():
            return W, True
        f0 = _objective(X, Y, W, lam, reg)
        g0 = G.abs().amax(0)
        step = torch.ones(m, device=X.device)
        for _ in range(30):
            Wn = W - step * delta
            f1 = _objective(X, Y, Wn, lam, reg)
            # Армихо; у оптимума разность целей во float32 тонет в округлении -- тогда шаг принимается, если цель не
            # выросла сверх точности и заметно упала норма градиента (она считается по остаткам точно)
            armijo = f1 <= f0 - 1e-4 * step * dec
            g1 = (X.t() @ (torch.sigmoid(X @ Wn) - Y) + lam * reg[:, None] * Wn).abs().amax(0)
            grad_ok = (g1 <= 0.9 * g0) & (f1 <= f0 + 1e-6 * f0.abs())
            bad = ~(armijo | grad_ok) & active
            if not bad.any():
                break
            step = torch.where(bad, step / 2, step)
        W = W - step * delta * active.float()
    return W, False


def add_bias(X):
    return torch.cat([X, torch.ones(len(X), 1, device=X.device, dtype=X.dtype)], 1)


def ap_columns(S, Y):
    """AP каждого столбца (концепта); для столбцов без положительных -- nan."""
    o = S.argsort(0, descending=True)
    Ys = Y.gather(0, o)
    tp = Ys.cumsum(0)
    prec = tp / torch.arange(1, len(S) + 1, device=S.device)[:, None]
    pos = Ys.sum(0)
    ap = (prec * Ys).sum(0) / pos.clamp_min(1)
    ap[pos == 0] = float("nan")
    return ap
