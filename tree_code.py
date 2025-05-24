import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Находит оптимальный порог для разбиения вектора признака по критерию Джини.

    Критерий Джини определяется следующим образом:
    .. math::
        Q(R) = -\\frac {|R_l|}{|R|}H(R_l) -\\frac {|R_r|}{|R|}H(R_r),

    где:
    * :math:`R` — множество всех объектов,
    * :math:`R_l` и :math:`R_r` — объекты, попавшие в левое и правое поддерево соответственно.

    Функция энтропии :math:`H(R)`:
    .. math::
        H(R) = 1 - p_1^2 - p_0^2,

    где:
    * :math:`p_1` и :math:`p_0` — доля объектов класса 1 и 0 соответственно.

    Указания:
    - Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    - В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака.
    - Поведение функции в случае константного признака может быть любым.
    - При одинаковых приростах Джини нужно выбирать минимальный сплит.
    - Для оптимизации рекомендуется использовать векторизацию вместо циклов.

    Parameters
    ----------
    feature_vector : np.ndarray
        Вектор вещественнозначных значений признака.
    target_vector : np.ndarray
        Вектор классов объектов (0 или 1), длина `feature_vector` равна длине `target_vector`.

    Returns
    -------
    thresholds : np.ndarray
        Отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на
        два различных поддерева.
    ginis : np.ndarray
        Вектор со значениями критерия Джини для каждого порога в `thresholds`.
    threshold_best : float
        Оптимальный порог для разбиения.
    gini_best : float
        Оптимальное значение критерия Джини.

    """
    x = feature_vector
    y = target_vector
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    uniq = np.unique(xs)
    if uniq.size <= 1:
        return np.array([]), np.array([]), None, None
    thr = (uniq[:-1] + uniq[1:]) / 2
    ginis = []
    n = ys.size
    for t in thr:
        left = ys[xs < t]
        right = ys[xs >= t]
        if left.size == 0 or right.size == 0:
            continue
        p1 = left.mean()
        g1 = 1 - p1**2 - (1 - p1)**2
        p2 = right.mean()
        g2 = 1 - p2**2 - (1 - p2)**2
        ginis.append(-(left.size/n)*g1 - (right.size/n)*g2)
    if not ginis:
        return thr, np.array(ginis), None, None
    ginis = np.array(ginis)
    bi = np.argmax(ginis)
    return thr, ginis, thr[bi], ginis[bi]

class DecisionTree:
    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
    ):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth = 0):
        """
        Обучение узла дерева решений.

        Если все элементы в подвыборке принадлежат одному классу, узел становится терминальным.

        Parameters
        ----------
        sub_X : np.ndarray
            Подвыборка признаков.
        sub_y : np.ndarray
            Подвыборка меток классов.
        node : dict
            Узел дерева, который будет заполнен информацией о разбиении.

        """
        if np.all(sub_y == sub_y[0]):
            node["type"]  = "terminal"
            node["class"] = sub_y[0]
            return
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"]  = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        if self._min_samples_split and sub_y.size < self._min_samples_split:
            node["type"]  = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        best_f = best_thr = best_g = best_cat = None
        best_split = None

        for j in range(sub_X.shape[1]):
            ft = self._feature_types[j]
            if ft == "real":
                vals = sub_X[:, j]
            else:
                cnt  = Counter(sub_X[:, j])
                clk  = Counter(sub_X[sub_y == 1, j])
                order = sorted(cnt, key=lambda k: clk.get(k, 0)/cnt[k])
                mapper = {c:i for i,c in enumerate(order)}
                vals = np.vectorize(mapper.get)(sub_X[:, j])

            if len(np.unique(vals)) <= 1:
                continue

            res = find_best_split(vals, sub_y)
            if not isinstance(res, tuple):
                continue
            _, _, thr, g = res
            if thr is None:
                continue

            left_cnt  = (vals < thr).sum()
            right_cnt = sub_y.size - left_cnt
            if self._min_samples_leaf and (left_cnt < self._min_samples_leaf or right_cnt < self._min_samples_leaf):
                continue

            if best_g is None or g > best_g:
                best_f     = j
                best_thr   = thr
                best_g     = g
                best_split = vals < thr
                if ft != "real":
                    best_cat = [k for k,v in mapper.items() if v < thr]

        if best_f is None:
            node["type"]  = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"]          = "nonterminal"
        node["feature_split"] = best_f
        if self._feature_types[best_f] == "real":
            node["threshold"] = best_thr
        else:
            node["categories_split"] = best_cat

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[ best_split], sub_y[ best_split], node["left_child"],  depth + 1)
        self._fit_node(sub_X[~best_split], sub_y[~best_split], node["right_child"], depth + 1)


    def _predict_node(self, x, node):
        """
        Рекурсивное предсказание класса для одного объекта по узлу дерева решений.

        Если узел терминальный, возвращается предсказанный класс.
        Если узел не терминальный, выборка передается в соответствующее поддерево для дальнейшего предсказания.

        Parameters
        ----------
        x : np.ndarray
            Вектор признаков одного объекта.
        node : dict
            Узел дерева решений.

        Returns
        -------
        int
            Предсказанный класс объекта.
        """
        if node["type"] == "terminal":
            return node["class"]
        f = node["feature_split"]
        if "threshold" in node:
            nxt = node["left_child"] if x[f] < node["threshold"] else node["right_child"]
        else:
            nxt = node["left_child"] if x[f] in node["categories_split"] else node["right_child"]
        return self._predict_node(x, nxt)

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)