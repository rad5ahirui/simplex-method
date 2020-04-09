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

status = ('Optimal', 'Feasible', 'Unbounded', 'Infeasible', 'Continue')
OPTIMAL = 0
FEASIBLE = 1
UNBOUNDED = 2
INFEASIBLE = 3
CONTINUE = 4

class SimplexMethod(object):
    def __init__(self, A, b, c, nonneg_idx, mins):
        self._orig_m, self._orig_n = A.shape
        self._A, self._b, self._c, self._is_nonneg, self._neg_idx, self._M =\
            self._rewrite_matrix(A, b, c, nonneg_idx, mins)
        self._m, self._n = self._A.shape
        self._status = CONTINUE
        self._tableau = None

    @property
    def status(self):
        return self._status

    def maximize(self):
        self._status, self._tableau = self._phase1()
        if self.status not in (OPTIMAL, FEASIBLE):
            return None, None
        self._status, self._tableau = self._simplex(self._tableau)
        if self.status != OPTIMAL:
            return None, None
        maximum = self._tableau[-1, -1]
        optimum = self._get_optimum(self._tableau)
        return maximum, self._original_vertex(optimum)

    def _rewrite_matrix(self, A, b, c, nonneg_idx, mins):
        m, n = A.shape
        M = Matrix(mins)
        b = b - A * M
        is_nonneg = [False] * n
        for i in nonneg_idx:
            is_nonneg[i] = True
        neg_idx = {}
        for i in range(n):
            if not is_nonneg[i]:
                A = A.row_join(-A.col(i))
                c = c.row_join(-c.col(i))
                neg_idx[i] = A.shape[1] - 1
        return A, b, c, is_nonneg, neg_idx, M

    def _original_vertex(self, x):
        y = Matrix.zeros(self._orig_n, 1)
        for i in range(self._orig_n):
            if self._is_nonneg[i]:
                y[i] = x[i]
            else:
                y[i] = x[i] - x[self._neg_idx[i]]
        return y + self._M

    def _get_optimum(self, tableau):
        m, n = tableau.shape
        nm1 = n - 1
        found = [False] * nm1
        optimum = []
        for i in range(nm1):
            idx = self._unit_vector_index(tableau[:, i])
            if idx != -1 and not found[idx]:
                found[idx] = True
                optimum.append(tableau[idx, -1])
            else:
                optimum.append(0)
        return Matrix(optimum)

    def _unit_vector_index(self, v):
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

    def _split_matrix(self):
        m, n = self._A.shape
        mp, mpp = 0, 0
        Ap = None
        bp = None
        App = None
        bpp = None
        for i in range(m):
            if self._b[i, 0] >= 0:
                if mp == 0:
                    Ap = self._A.row(i)
                    bp = self._b.row(i)
                else:
                    Ap = Ap.col_join(self._A.row(i))
                    bp = bp.col_join(self._b.row(i))
                mp += 1
            else:
                if mpp == 0:
                    App = self._A.row(i)
                    bpp = self._b.row(i)
                else:
                    App = App.col_join(self._A.row(i))
                    bpp = bpp.col_join(self._b.row(i))
                mpp += 1
        return Ap, App, bp, bpp

    def _initial_tableau(self, Ap, App, bp, bpp):
        mp, n = Ap.shape
        e_mp = Matrix.eye(mp)
        if App is None:
            mpp = 0
            tableau = Ap.row_join(e_mp).row_join(bp)
            tableau = tableau.col_join(Matrix.zeros(1, n + mp + 1))
        else:
            mpp = App.shape[0]
            z_mp = Matrix.zeros(mp, mpp)
            z_mpp = Matrix.zeros(mpp, mp)
            e_mpp = Matrix.eye(mpp)
            o = Matrix.ones(1, mpp)
            tableau = Ap.row_join(z_mp).row_join(e_mp).row_join(z_mp).row_join(bp)
            tableau = tableau.col_join((-App).row_join(-e_mpp).row_join(z_mpp)\
                                       .row_join(e_mpp).row_join(-bpp))
            tableau = tableau.col_join((o * App).row_join(o)\
                                       .row_join(Matrix.zeros(1, mp + mpp + 1)))
        return tableau, mp, mpp

    def _simplex(self, tableau):
        p, q = tableau.shape
        qm1 = q - 1
        while True:
            # Choose a column
            col = -1
            for i in range(qm1):
                if tableau[-1, i] < 0:
                    col = i
                    break
            if col == -1:
                return OPTIMAL, tableau
            # Choose a row
            row = -1
            for i in range(self._m):
                if tableau[i, col] > 0:
                    min_ = tableau[i, -1] / tableau[i, col]
                    row = i
                    break
            if row == -1:
                return UNBOUNDED, tableau
            for i in range(row + 1, self._m):
                if tableau[i, col] > 0:
                    k = tableau[i, -1] / tableau[i, col]
                    if k < min_:
                        min_ = k
                        row = i
            # Elimination
            tableau[row, :] /= tableau[row, col]
            for i in range(p):
                if i != row:
                    tableau[i, :] -= tableau[i, col] * tableau.row(row)

    def _modify_tableau(self, tableau, mp, mpp, n):
        m = mp + mpp
        col = n + m
        for i in range(col, col + mpp):
            idx = self._unit_vector_index(tableau[:, col])
            if idx != -1:
                tableau[idx, :] *= -1
            tableau.col_del(col) # inefficent?
        tableau[-1, :] = (-self._c).row_join(Matrix.zeros(1, m + 1))
        found = [False] * m
        for i in range(n):
            idx = self._unit_vector_index(tableau[:-1, i])
            if idx != -1 and not found[idx]:
                tableau[-1, :] += self._c[i] * tableau.row(idx)
                found[idx] = True
        return tableau

    def _phase1(self):
        m, n = self._A.shape
        Ap, App, bp, bpp = self._split_matrix()
        if Ap is None:
            return INFEASIBLE, None
        tableau, mp, mpp = self._initial_tableau(Ap, App, bp, bpp)
        status, tableau = self._simplex(tableau)
        if status != OPTIMAL:
            return INFEASIBLE, None
        return FEASIBLE, self._modify_tableau(tableau, mp, mpp, n)

def main():
    from sympy import Matrix
    A = Matrix([[1, 1, 1],
                [-2, -1, 1],
                [0, 1, -1]])
    b = Matrix([40, -10, -10])
    c = Matrix([[2, 3, 1]])
    nonneg_idx = [0, 1, 2]
    mins = [0, 0, 0]
    solver = SimplexMethod(A, b, c, nonneg_idx, mins)
    maximum, optimum = solver.maximize()
    print(status[solver.status])
    pprint(maximum)
    pprint(optimum)

if __name__ == '__main__':
    main()
