import matplotlib.pyplot as plt
import numpy as np



class TernaryPlot:

    def __init__(self, grid_intervals=10):
        self.grid_intervals = grid_intervals
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self._draw_triangle()
        self._draw_grid_lines()
        self.ax.set_aspect('equal')
        self.ax.axis('off')

    def _draw_triangle(self):
        triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
        self.ax.plot(triangle[:, 0], triangle[:, 1], 'k')
        
        A = np.array([0, 0])
        B = np.array([1, 0])
        C = np.array([0.5, np.sqrt(3)/2])

        t = 0.1
        offset = 0.05

        AB = A + t * (B - A)
        self.ax.text(AB[0], AB[1] - offset, 'Water (wt %)', fontsize=11,
                    ha='center', va='center', rotation=0)

        BC = B + t * (C - B)
        self.ax.text(BC[0] + offset, BC[1], 'Thymol (wt %)', fontsize=11,
                    ha='center', va='center', rotation=-60)

        CA = C + t * (A - C)
        self.ax.text(CA[0] - offset + 0.01, CA[1] + 0.03, 'Ethanol (wt %)', fontsize=11,
                    ha='center', va='center', rotation=60)

    def _draw_grid_lines(self):
        n = self.grid_intervals
        for i in range(1, n):
            frac = i / n
            # Lines orthogonal to A-axis
            x1, y1 = self.ternary_to_cartesian([frac], [1 - frac], [0])
            x2, y2 = self.ternary_to_cartesian([frac], [0], [1 - frac])
            self.ax.plot([x1[0], x2[0]], [y1[0], y2[0]], linestyle='dotted', color='gray', linewidth=0.7)

            # Lines orthogonal to B-axis
            x1, y1 = self.ternary_to_cartesian([0], [frac], [1 - frac])
            x2, y2 = self.ternary_to_cartesian([1 - frac], [frac], [0])
            self.ax.plot([x1[0], x2[0]], [y1[0], y2[0]], linestyle='dotted', color='gray', linewidth=0.7)

            # Lines orthogonal to C-axis
            x1, y1 = self.ternary_to_cartesian([1 - frac], [0], [frac])
            x2, y2 = self.ternary_to_cartesian([0], [1 - frac], [frac])
            self.ax.plot([x1[0], x2[0]], [y1[0], y2[0]], linestyle='dotted', color='gray', linewidth=0.7)

    def ternary_to_cartesian(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        total = a + b + c
        x = 0.5 * (2 * b + c) / total
        y = (np.sqrt(3) / 2) * c / total
        return x, y
    
    def plot_points(self, abc_list, label=None, marker='o', color='blue'):
        for abc in abc_list:
            a, b, c = float(abc[0]), float(abc[1]), float(abc[2])
            x, y = self.ternary_to_cartesian([a], [b], [c])
            self.ax.plot(x, y, marker=marker, markersize=3, color=color, label=label, linestyle='')
        if label:
            self.ax.legend()

    def _draw_rotated_marker(self, x, y, angle_deg, size, **kwargs):
        import numpy as np
        angle_rad = np.deg2rad(angle_deg)
        dx = (size / 2) * np.cos(angle_rad)
        dy = (size / 2) * np.sin(angle_rad)
        self.ax.plot([x - dx, x + dx], [y - dy, y + dy], **kwargs)

    def add_border_labels(self):
        n = 5
        for i in range(1, n):
            frac = i / n
            size = 0.01  # Line segment length for "marker"

            a, b, c = 1 - frac, frac, 0
            x, y = self.ternary_to_cartesian([a], [b], [c])
            self._draw_rotated_marker(x[0], y[0], angle_deg=-60, size=size, color='black', linewidth=1)
            self.ax.text(x[0] + 0.01, y[0] - 0.03, f"{round(a*100)}%",rotation =-60, ha='center', va='center', fontsize=8)

            a, b, c = 0, 1 - frac, frac
            x, y = self.ternary_to_cartesian([a], [b], [c])
            self._draw_rotated_marker(x[0], y[0], angle_deg=60, size=size, color='black', linewidth=1)
            self.ax.text(x[0] + 0.001, y[0] + 0.03, f"{round(b*100)}%", rotation=60, ha='left', va='center', fontsize=8)

            a, b, c = frac, 0, 1 - frac
            x, y = self.ternary_to_cartesian([a], [b], [c])
            self._draw_rotated_marker(x[0], y[0], angle_deg=0, size=size, color='black', linewidth=1)
            self.ax.text(x[0] - 0.01, y[0], f"{round(c*100)}%", rotation=0, ha='right', va='center', fontsize=8)

    def add_component_edge_markers(self, a, b, c, col, size=0.01):
        x0, y0 = self.ternary_to_cartesian([a], [b], [c])

        # ethanol
        x1, y1 = self.ternary_to_cartesian([a], [0], [c])
        self._draw_rotated_marker(x1[0], y1[0], angle_deg=0, size=size, color=col , linewidth=1)
        self.ax.plot([x0[0], x1[0]], [y0[0], y1[0]], linestyle='dotted', color=col, linewidth=0.7)

        # thymol
        norm = b + c
        x1, y1 = self.ternary_to_cartesian([0], [b*norm], [c]) 
        self._draw_rotated_marker(x1[0], y1[0], angle_deg=60, size=size, color=col , linewidth=1)
        self.ax.plot([x0[0], x1[0]], [y0[0], y1[0]], linestyle='dotted', color=col, linewidth=0.7)

        # water
        norm = a + b
        x1, y1 = self.ternary_to_cartesian([a], [b*11], [0]) 
        self._draw_rotated_marker(x1[0], y1[0], angle_deg=-60, size=size, color=col , linewidth=1)
        self.ax.plot([x0[0], x1[0]], [y0[0], y1[0]], linestyle='dotted', color=col, linewidth=0.7)
    
    def add_component_label(self, a, b, c, text, col, dx=-0.03, dy=0.03, size=12):
        #ethanol label
        x1, y1 = self.ternary_to_cartesian([a], [0], [c])
        self.ax.text(x1[0] - 0.017, y1[0] - 0.001, f'{round(c*100, 2)}%', color=col, fontsize = size)

        #water label
        x2, y2 = self.ternary_to_cartesian([a], [b*11], [0])
        self.ax.text(x2[0] + 0.001, y2[0] - 0.018, f'{round(a*100, 2)}%', color=col, fontsize = size, rotation = -60)

    def draw_zoom_region(self, a_range, b_range, c_range, padding = 0.01):
        a_vals = [a_range[0], a_range[1], a_range[1], a_range[0]]
        b_vals = [b_range[0], b_range[0], b_range[1], b_range[1]]
        c_vals = [c_range[0], c_range[1], c_range[0], c_range[1]]

        total = np.array(a_vals) + np.array(b_vals) + np.array(c_vals)
        a_vals = np.array(a_vals) / total
        b_vals = np.array(b_vals) / total
        c_vals = np.array(c_vals) / total

        x, y = self.ternary_to_cartesian(a_vals, b_vals, c_vals)
        col = 'black'
        line_style = 'dashed'

        self.ax.plot([min(x)- padding, max(x)], [min(y)- padding, min(y) - padding], linestyle = line_style, color=col, linewidth=0.9)
        self.ax.plot([min(x) - padding, min(x) - padding], [min(y) - padding, max(y) + padding], linestyle=line_style, color=col, linewidth=0.9) 
        self.ax.plot([min(x) - padding, max(x)], [max(y) + padding, max(y) + padding], linestyle=line_style, color=col, linewidth=0.9) 
        self.ax.plot([max(x), max(x)], [max(y) + padding, min(y) - padding], linestyle=line_style, color=col, linewidth=0.9) #

    def zoom_to_region(self, a_range, b_range, c_range, padding=0.01):
        a_vals = [a_range[0], a_range[1], a_range[1], a_range[0]]
        b_vals = [b_range[0], b_range[0], b_range[1], b_range[1]]
        c_vals = [c_range[0], c_range[1], c_range[0], c_range[1]]

        # Normalize ternary coordinates to ensure they sum to 1
        total = np.array(a_vals) + np.array(b_vals) + np.array(c_vals)
        a_vals = np.array(a_vals) / total
        b_vals = np.array(b_vals) / total
        c_vals = np.array(c_vals) / total

        x, y = self.ternary_to_cartesian(a_vals, b_vals, c_vals)

        self.ax.set_xlim(min(x) - padding, max(x) + padding)
        self.ax.set_ylim(min(y) - padding, max(y) + padding)



    def showandsave(self, path):
        plt.tight_layout()
        plt.savefig(path, dpi = 300)
        plt.show()

#thymol ethanol: 0.099919
def make_points(thymol_percent):
    wt_thymol = thymol_percent / 100
    wt_ethanol = 9.9919 * wt_thymol
    wt_water = 1 - (wt_ethanol + wt_thymol)

    #water, thymol, ethanol
    return [wt_water, wt_thymol, wt_ethanol]


def plot_1():
    thymol_percentages = [0.14, 0.18, 0.27, 0.39, 0.51, 0.81]
    points = [make_points(x) for x in thymol_percentages]
    cols = ['#88CCEE', '#44AA99', '#117733', '#DDCC77', '#EE9966', '#CC6677', '#CC6677']

    # First Plot
    tern_plot = TernaryPlot(grid_intervals=10)

    for point, col in zip(points, cols):
        tern_plot.plot_points([point], label=f'{np.round(point[1]*100, 2)} wt. %', color=col)
        #tern_plot.plot_points([point], color=col)
        tern_plot.add_component_edge_markers(*point, col)
        #tern_plot.add_component_label(*point, 'test', col)

    """tern_plot.zoom_to_region(
        a_range=(0.85, 1.0),  # Water
        b_range=(0.0, 0.05),  # Thymol
        c_range=(0.0, 0.1)    # Ethanol
    )"""

    tern_plot.draw_zoom_region(
        a_range=(0.85, 1.0),  # Water
        b_range=(0.0, 0.05),  # Thymol
        c_range=(0.0, 0.1)    # Ethanol
    )

    tern_plot.add_border_labels()
    tern_plot.showandsave('/home/elias/proj/_photon_correlation/ternary.png')

def plot_2():
    thymol_percentages = [0.19, 0.26, 0.40, 0.76]
    points = [make_points(x) for x in thymol_percentages]
    cols = ['#88CCEE', '#44AA99', '#117733', '#DDCC77', '#EE9966', '#CC6677', '#CC6677']

    # First Plot
    tern_plot = TernaryPlot(grid_intervals=10)

    for point, col in zip(points, cols):
        tern_plot.plot_points([point], label=f'{np.round(point[1]*100, 2)} wt. %', color=col)
        #tern_plot.plot_points([point], color=col)
        tern_plot.add_component_edge_markers(*point, col)
        #tern_plot.add_component_label(*point, 'test', col)

    """tern_plot.zoom_to_region(
        a_range=(0.85, 1.0),  # Water
        b_range=(0.0, 0.05),  # Thymol
        c_range=(0.0, 0.1)    # Ethanol
    )"""

    tern_plot.draw_zoom_region(
        a_range=(0.85, 1.0),  # Water
        b_range=(0.0, 0.05),  # Thymol
        c_range=(0.0, 0.1)    # Ethanol
    )

    tern_plot.add_border_labels()
    tern_plot.showandsave('/home/elias/proj/_photon_correlation/ternary.png')


def plot_3():
    thymol_percentages = [0.19, 0.33, 0.45]
    points = [make_points(x) for x in thymol_percentages]
    cols = ['#44AA99','#117733', '#DDCC77']

    # First Plot
    tern_plot = TernaryPlot(grid_intervals=10)

    for point, col in zip(points, cols):
        tern_plot.plot_points([point], label=f'{np.round(point[1]*100, 2)} wt. %', color=col)
        #tern_plot.plot_points([point], color=col)
        tern_plot.add_component_edge_markers(*point, col)
        #tern_plot.add_component_label(*point, 'test', col)

    tern_plot.draw_zoom_region(
        a_range=(0.85, 1.0),  # Water
        b_range=(0.0, 0.05),  # Thymol
        c_range=(0.0, 0.1)    # Ethanol
    )
    tern_plot.add_border_labels()
    tern_plot.showandsave('/home/elias/proj/_photon_correlation/ternary.png')

plot_3()