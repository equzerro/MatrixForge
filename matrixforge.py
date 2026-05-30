"""
MatrixForge 
Символьное дифференцирование + решатель разреженных СЛАУ
Без внешних зависимостей (только stdlib)
"""

import re
import math


# ============================================================
# ЧАСТЬ 1. ПАРСЕР ВЫРАЖЕНИЙ (рекурсивный спуск)
# ============================================================

class Node:
    """Узел AST"""
    def __init__(self, type_, value=None, left=None, right=None):
        self.type = type_      # num, var, +, -, *, /, ^, func
        self.value = value
        self.left = left
        self.right = right


class Parser:
    """Парсер математических выражений методом рекурсивного спуска"""
    FUNCS = {'sin', 'cos', 'exp', 'ln', 'tan'}

    def __init__(self, text):
        self.tokens = self._tokenize(text)
        self.pos = 0

    def _tokenize(self, text):
        text = text.replace(' ', '')
        pattern = r'(\d+\.?\d*|[a-zA-Z]+|[\+\-\*/\^\(\)])'
        return re.findall(pattern, text)

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self):
        return self._expr()

    def _expr(self):           # сложение и вычитание
        node = self._term()
        while self._peek() in ('+', '-'):
            op = self._next()
            node = Node(op, left=node, right=self._term())
        return node

    def _term(self):           # умножение и деление
        node = self._power()
        while self._peek() in ('*', '/'):
            op = self._next()
            node = Node(op, left=node, right=self._power())
        return node

    def _power(self):          # возведение в степень
        node = self._factor()
        if self._peek() == '^':
            self._next()
            node = Node('^', left=node, right=self._power())
        return node

    def _factor(self):         # числа, переменные, скобки, функции
        tok = self._peek()
        if tok == '(':
            self._next()
            node = self._expr()
            self._next()       # пропускаем ')'
            return node
        if tok == '-':         # унарный минус
            self._next()
            return Node('-', left=Node('num', 0), right=self._factor())
        if tok in self.FUNCS:
            self._next()
            self._next()       # '('
            arg = self._expr()
            self._next()       # ')'
            return Node('func', value=tok, left=arg)
        if re.match(r'^\d', tok):
            self._next()
            return Node('num', float(tok))
        self._next()
        return Node('var', tok)


# ============================================================
# ЧАСТЬ 2. СИМВОЛЬНОЕ ДИФФЕРЕНЦИРОВАНИЕ
# ============================================================

def differentiate(node, var='x'):
    """Производная узла AST по переменной var"""
    if node.type == 'num':
        return Node('num', 0)
    if node.type == 'var':
        return Node('num', 1 if node.value == var else 0)

    if node.type in ('+', '-'):
        return Node(node.type,
                    left=differentiate(node.left, var),
                    right=differentiate(node.right, var))

    if node.type == '*':                # (uv)' = u'v + uv'
        return Node('+',
            left=Node('*', left=differentiate(node.left, var), right=node.right),
            right=Node('*', left=node.left, right=differentiate(node.right, var)))

    if node.type == '/':                # (u/v)' = (u'v - uv')/v^2
        u, v = node.left, node.right
        num = Node('-',
            left=Node('*', left=differentiate(u, var), right=v),
            right=Node('*', left=u, right=differentiate(v, var)))
        den = Node('^', left=v, right=Node('num', 2))
        return Node('/', left=num, right=den)

    if node.type == '^':                # x^n -> n*x^(n-1)
        base, exp = node.left, node.right
        if exp.type == 'num':
            new_exp = Node('num', exp.value - 1)
            return Node('*',
                left=Node('*', left=Node('num', exp.value),
                          right=Node('^', left=base, right=new_exp)),
                right=differentiate(base, var))

    if node.type == 'func':             # цепное правило
        inner = differentiate(node.left, var)
        if node.value == 'sin':
            d = Node('func', value='cos', left=node.left)
        elif node.value == 'cos':
            d = Node('-', left=Node('num', 0),
                     right=Node('func', value='sin', left=node.left))
        elif node.value == 'exp':
            d = Node('func', value='exp', left=node.left)
        elif node.value == 'ln':
            d = Node('/', left=Node('num', 1), right=node.left)
        else:
            d = Node('num', 0)
        return Node('*', left=d, right=inner)

    return Node('num', 0)


def to_str(node):
    """Преобразование AST обратно в строку"""
    if node is None:
        return ''
    if node.type == 'num':
        v = node.value
        return str(int(v)) if v == int(v) else str(v)
    if node.type == 'var':
        return node.value
    if node.type == 'func':
        return f"{node.value}({to_str(node.left)})"
    return f"({to_str(node.left)} {node.type} {to_str(node.right)})"


# ============================================================
# ЧАСТЬ 3. РЕШАТЕЛЬ РАЗРЕЖЕННЫХ СЛАУ (метод сопряжённых градиентов)
# ============================================================

class SparseMatrix:
    """Разреженная матрица в формате CSR (хранит только ненулевые)"""
    def __init__(self, n):
        self.n = n
        self.data = {}   # (i, j) -> value

    def set(self, i, j, value):
        if value != 0:
            self.data[(i, j)] = value

    def multiply(self, vec):
        """Умножение матрицы на вектор: O(nnz)"""
        result = [0.0] * self.n
        for (i, j), val in self.data.items():
            result[i] += val * vec[j]
        return result


def conjugate_gradient(A, b, tol=1e-10, max_iter=1000):
    """Метод сопряжённых градиентов для решения Ax=b"""
    n = A.n
    x = [0.0] * n
    r = b[:]                       # невязка
    p = r[:]
    rs_old = sum(ri * ri for ri in r)

    for it in range(max_iter):
        Ap = A.multiply(p)
        alpha = rs_old / sum(p[i] * Ap[i] for i in range(n))
        x = [x[i] + alpha * p[i] for i in range(n)]
        r = [r[i] - alpha * Ap[i] for i in range(n)]
        rs_new = sum(ri * ri for ri in r)
        if math.sqrt(rs_new) < tol:
            return x, it + 1
        p = [r[i] + (rs_new / rs_old) * p[i] for i in range(n)]
        rs_old = rs_new
    return x, max_iter


# ============================================================
# CLI-ИНТЕРФЕЙС
# ============================================================

def main():
    print("=" * 50)
    print("       MatrixForge v0.1-Demo")
    print("=" * 50)
    while True:
        print("\n1 — Символьная производная")
        print("2 — Решить разреженную СЛАУ")
        print("0 — Выход")
        choice = input(">>> ").strip()

        if choice == '1':
            expr = input("Введите выражение f(x): ")
            try:
                tree = Parser(expr).parse()
                deriv = differentiate(tree, 'x')
                print(f"f'(x) = {to_str(deriv)}")
            except Exception as e:
                print(f"Ошибка разбора: {e}")

        elif choice == '2':
            # Пример: трёхдиагональная система 4x4
            print("Демо: решаем систему 4x4 (диагональ=4, соседи=-1)")
            n = 4
            A = SparseMatrix(n)
            for i in range(n):
                A.set(i, i, 4)
                if i > 0:
                    A.set(i, i - 1, -1)
                if i < n - 1:
                    A.set(i, i + 1, -1)
            b = [1, 2, 3, 4]
            x, iters = conjugate_gradient(A, b)
            print(f"Решение: {[round(v, 4) for v in x]}")
            print(f"Итераций: {iters}")

        elif choice == '0':
            print("До свидания!")
            break
        else:
            print("Неверный выбор.")


if __name__ == "__main__":
    main()