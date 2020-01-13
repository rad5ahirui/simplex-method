#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (C) 2019 Ahirui Otsu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from sympy import Matrix

status = ('Optimal', 'Unbounded', 'Infeasible')
OPTIMAL = 0
UNBOUNDED = 1
INFEASIBLE = 2

def _forward_elimination(A, b):
    m, n = A.shape
    p = min(m, n)
    rankA = 0
    r = min(m, n)
    for k in range(r):
        # Pivoting
        max_ = abs(A[k, k])
        max_idx = k
        for i in range(k + 1, m):
            e = abs(A[i, k])
            if e > max_:
                max_idx = i
                max_ = e
        if max_ == 0:
            break
        elif max_idx != k:
            A.row_swap(k, max_idx)
            b.row_swap(k, max_idx)
        # Forward elimination
        for i in range(k + 1, m):
            e = A[i, k] / A[k, k]
            A[i, :] = A.row(i) - e * A.row(k)
            b[i, :] = b.row(i) - e * b.row(k)
        rankA += 1
    rankAb = rankA
    if rankA < r and max(b[rankA:, :]) != 0:
        rankAb += 1
    if rankAb < m:
        b[rankAb:, :] = Matrix.zeros(m - rankAb, 1)
    return rankA, rankAb

def _make_tableau_phase1(Ap, App, bp, bpp):
    mp, n = Ap.shape
    mpp = App.shape[0]
    z_mp = Matrix.zeros(mp, mpp)
    e_mp = Matrix.eye(mp)
    z_mpp = Matrix.zeros(mpp, mp)
    e_mpp = Matrix.eye(mpp)
    o = Matrix.ones(1, mpp)
    tab = Ap.row_join(z_mp).row_join(e_mp).row_join(z_mp).row_join(bp)
    tab = tab.col_join((-App).row_join(-e_mpp).row_join(z_mpp).row_join(e_mpp).row_join(-bpp))
    tab = tab.col_join((o * App).row_join(o).row_join(Matrix.zeros(1, mp + mpp + 1)))
    return tab

def _simplex(tab):
    m, n = tab.shape
    mm1 = m - 1
    nm1 = n - 1
    while True:
        # Choose a column
        col = -1
        for i in range(nm1):
            if tab[-1, i] < 0:
                col = i
                break
        if col == -1:
            return OPTIMAL
        # Choose a row
        row = -1
        for i in range(mm1):
            if tab[i, col] > 0:
                min_ = tab[i, -1] / tab[i, col]
                row = i
                break
        if row == -1:
            return UNBOUNDED
        for i in range(row, mm1):
            if tab[i, col] > 0:
                k = tab[i, -1] / tab[i, col]
                if k < min_:
                    min_ = k
                    row = i
        # Elimination
        tab[row, :] /= tab[row, col]
        for i in range(m):
            if i != row:
                tab[i, :] -= tab[i, col] * tab[row, :]

def _unit_vector_index(v):
    n = v.shape[0]
    idx = -1
    all_zeros = True
    for i in range(n):
        e = v[i, 0]
        if e == 1 and all_zeros:
            all_zeros = False
            idx = i
        elif e != 0:
            idx = -1
            break
    return idx

def _modify_tableau(tab, mp, mpp, n, c):
    m = mp + mpp
    col = n + m
    for i in range(col, col + mpp):
        idx = _unit_vector_index(tab[:-1, col])
        if idx != -1:
            tab[idx, :] *= -1
        tab.col_del(col) # inefficent?
    tab[-1, :] = (-c).row_join(Matrix.zeros(1, m + 1))
    found = [False] * m
    for i in range(n):
        idx = _unit_vector_index(tab[:-1, i])
        if idx != -1 and not found[idx]:
            tab[-1, :] += c[i] * tab[idx, :]
            found[idx] = True

def _get_basic_solution(tab):
    nm1 = tab.shape[1] - 1
    found = [False] * nm1
    basic = []
    nonbasic = []
    x = []
    for i in range(nm1):
        idx = _unit_vector_index(tab[:-1, i])
        if idx != -1 and not found[idx]:
            found[idx] = True
            basic.append(i)
            x.append(tab[idx, -1])
        else:
            nonbasic.append(i)
    return Matrix(x), basic, nonbasic

def _get_optimum(tab, m, n):
    basic_sol, basis_indices, nonbasis_indices = _get_basic_solution(tab)
    if nonbasis_indices[0] < n:
        # Are there any more efficient ways?
        B = tab[:-1, nonbasis_indices]
        d = tab[:-1, -1] - tab[:-1, basis_indices] * basic_sol
        r, _ = _forward_elimination(B, d)
        B = B[:r, :r]
        d = B[:r, 0]
        nonbasic_sol = B.inv() * d
        if n - r > 0:
            # The system of equation has infinitely many solutions
            nonbasic_sol = nonbasic_sol.col_join(Matrix.zeros(n - r, 1))
        i = 0 # basic_sol
        j = 0 # nonbasic_sol
        x_list = []
        for cnt in range(n):
            if i < m:
                if j < n:
                    if basis_indices[i] < nonbasis_indices[j]:
                        x_list.append(basic_sol[i])
                        i += 1
                    else:
                        x_list.append(nonbasic_sol[i])
                        j += 1
                else: # j == n
                    x_list.append(basic_sol[i])
                    i += 1
            else: # j < n
                x_list.append(nonbasic_sol[i])
                j += 1
        x = Matrix(x_list)
    elif m > n:
        x = basic_sol[:n, 0]
    else:
        x = basic_sol
    return x

def _get_form(A, b, c):
    m, n = A.shape
    mp, mpp = 0, 0
    App = None
    bpp = None
    for i in range(m):
        if b[i, 0] >= 0:
            if mp == 0:
                Ap = A.row(i)
                bp = b.row(i)
            else:
                Ap = Ap.col_join(A.row(i))
                bp = bp.col_join(b.row(i))
            mp += 1
        else:
            if mpp == 0:
                App = A.row(i)
                bpp = b.row(i)
            else:
                App = App.col_join(A.row(i))
                bpp = bpp.col_join(b.row(i))
            mpp += 1
    return mpp == 0, Ap, App, bp, bpp

# returns (status, max P, argmax_x P),
# where P = { cx: Ax <= b, x >= 0 }.
# status = {OPTIMAL, UNBOUNDED, INFEASIBLE}
def maximize(A, b, c):
    m, n = A.shape
    # convert LP to the form
    # max{ cx: Ap * x <= bp, App * x <= bpp, x >= 0 }
    # with b' >= 0 and b'' < 0.
    vertex_found, Ap, App, bp, bpp = _get_form(A, b, c)
    # Solve LP
    if vertex_found:
        mp = Ap.shape[0]
        # Make a tableau
        tab = Ap.row_join(Matrix.eye(mp)).row_join(bp)
        tab = tab.col_join((-c).row_join(Matrix.zeros(1, mp + 1)))
    else:
        # Two-phase simplex
        # Phase 1
        mpp = App.shape[0]
        tab = _make_tableau_phase1(Ap, App, bp, bpp)
        res = _simplex(tab)
        if res != OPTIMAL or tab[-1, -1] < Matrix.ones(1, mpp).dot(bpp):
            return INFEASIBLE, None, None
        _modify_tableau(tab, mp, mpp, n, c)
        # Phase 2
    res = _simplex(tab)
    if res != OPTIMAL:
        return UNBOUNDED, None, None
    x = _get_optimum(tab, m, n)
    return OPTIMAL, tab[-1, -1], x

def main():
    from sympy import pprint
    c = Matrix([[1, 1]])
    A = Matrix([[2, 3],
                [1, -1]])
    b = Matrix([12, 2])
    print('Maximize: cx')
    print('Subject to: Ax <= b, x >= 0')
    print('c =')
    pprint(c)
    print('A =')
    pprint(A)
    print('b =')
    pprint(b)
    res, opt, x = maximize(A, b, c)
    print()
    print(f'Status: {status[res]}')
    print(f'Maximum: {opt}')
    print(f'Optimum:')
    pprint(x)

if __name__ == '__main__':
    main()
