# -1- Лабораторная работа 2 на тему "Задача N тел: гравитационное взаимодействие".
Код моделирует задачу N тел - эволюцию системы из 500 гравитационных частиц во времени (кол-во частиц можно варьировать),
с использованием двух ключевых численных методов: интегратора Верле и дерева Барнса-Хата для быстрого расчета сил.
 - Метод Верле - численный метод моделирования уравнения движения
 - Метод Барнса-Хата - алгоритм расчета гравитационных сил, который снижает вычислительную сложность
# -2- Начальные условия
Моделируется тонкий диск радиуса R:
 - sqrt() - равномерное распределение по площади диска
 - z - N(0, 0.3) - малая толщина по вертикали
```
  R = 10.0
  angles = np.random.uniform(0, 2*np.pi, N)
  radii  = R * np.sqrt(np.random.uniform(0, 1, N))

  pos[:, 0] = radii * np.cos(angles)
  pos[:, 1] = radii * np.sin(angles)
  pos[:, 2] = np.random.normal(0, 0.3, N)
```
# -3- Прямой расчет ускорений O(N^2)
Попарный расчет всех N^2 взаимодействий.
```
def compute_acceleration_direct(pos, mass, G, soft):
    diff = pos[None, :, :] - pos[:, None, :]          # (N,N,3)
    dist2 = np.sum(diff**2, axis=2) + soft**2
    inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
    np.fill_diagonal(inv_dist3, 0.0)                  # исключить самодействие
    acc = G * np.einsum('ij,ijk->ik', mass[None, :] * inv_dist3, diff)
    return acc
```
 - Векторизация через (N,N,3) - массивы
 - fill_diagonal()=0 - частица не действует сама на себя
# -4- Дерево Барнса-Хата
Класс узла:
```
 class OctreeNode:
    __slots__ = ('center', 'half', 'mass', 'com', 'children', 'indices')
```
 - center, half - геометрия куба 
 - mass, com - суммарная масса и центр масс
 - children - дочерние узлы
 - indices - индексы частиц
 - _slots_ - экономия памяти
# -5- Сила от узла
```
 def tree_force(node, target_pos, target_mass, G, theta, soft):
    d = node.com - target_pos
    r2 = np.dot(d, d) + soft**2
    r  = np.sqrt(r2)
    s = node.half
    if node.is_leaf() or (s / r < theta):
        return G * target_mass * node.mass * d / (r2 * r)
    F = np.zeros(3)
    for child in node.children:
        F += tree_force(child, target_pos, target_mass, G, theta, soft)
    return F
```
Критерий Барнса-Хата "s/r < theta" - аппроксимируем весь узел одной точкой в центре масс. Если нет,
то рекурсивно спускаемся.
# -6- Диагностика
Контроль качества интегрирования. 0.5 в PE - компенсация двойного счета пар (i,j) и (j,i). soft входит 
и в расстояние, чтобы энергия соответствовала смягченной силе, что и в динамике.
```
def total_energy(pos, vel, mass, G, soft):
    KE = 0.5 * np.sum(mass[:, None] * vel**2)
    PE = -G * 0.5 * np.sum((mass[:, None] * mass[None, :]) / dist)
    return KE + PE
```
# -7- Интегратор Верле
Получаем pos_prev тейлоровским разложением назад:
```
acc = compute_acceleration(pos, mass, G, soft, use_tree, theta)
pos_prev = pos - dt * vel + 0.5 * dt**2 * acc   # обратный шаг для запуска
```
Три строки - вся динамика:
 1)Новая позиция по формуле Верле
 2)Скорость центральной разностью
 3)Новое ускорение (одно обращение к дереву на щаг)
```
for step in range(1, N_steps + 1):
    pos_new = 2.0 * pos - pos_prev + dt**2 * acc
    vel     = (pos_new - pos_prev) / (2.0 * dt)
    acc_new = compute_acceleration(pos_new, mass, G, soft, use_tree, theta)
    pos_prev, pos, acc = pos, pos_new, acc_new
```
# -8- Результат
Результатом кода является визуализация:
 - Траектория
```
for i in range(N):
    traj = np.array([h[i, :2] for h in history])
    ax.plot(traj[:, 0], traj[:, 1], lw=0.5, alpha=0.6, color=colors[i])
ax.scatter(pos[:, 0], pos[:, 1], s=8, c='red', label='конечное положение')
```
 - Энергии
```
steps_arr = np.arange(len(energies)) * save_every
ax.plot(steps_arr, energies, 'b-', lw=1.2)
```
 - Сохранение
```
plt.tight_layout()
plt.savefig('nbody_verlet_tree.png', dpi=120)
plt.show()
```
<img width="972" height="424" alt="image" src="https://github.com/user-attachments/assets/e0587e5f-600c-48ee-82ea-37843090a939" />
