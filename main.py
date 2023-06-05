import tkinter
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

matrix_entries = []
cost_matrix = None
assignments = {}


def hungarian_algorithm(cost_matrix, maximize=True):
    # Step 1: Subtract the minimum value from each row
    min_row_vals = np.min(cost_matrix, axis=1)
    cost_matrix -= min_row_vals[:, np.newaxis]

    # Step 2: Subtract the minimum value from each column
    min_col_vals = np.min(cost_matrix, axis=0)
    cost_matrix -= min_col_vals

    num_workers, num_jobs = cost_matrix.shape
    assignments = {}
    remaining_jobs = set(range(num_jobs))

    for _ in range(num_workers):
        max_value = -np.inf if maximize else np.inf
        max_worker = None
        max_job = None

        # Find the maximum/minimum value among the remaining unassigned workers and jobs
        for i in range(num_workers):
            if i not in assignments:
                for j in remaining_jobs:
                    if (cost_matrix[i, j] > max_value) if maximize else (cost_matrix[i, j] < max_value):
                        max_value = cost_matrix[i, j]
                        max_worker = i
                        max_job = j

        if (max_value == 0 and maximize) or (max_value == np.inf and not maximize):
            break

        # Assign the job to the worker
        assignments[max_worker] = max_job
        remaining_jobs.remove(max_job)

    # Assign the remaining unassigned workers to the remaining jobs
    unassigned_workers = [i for i in range(num_workers) if i not in assignments]
    for worker in unassigned_workers:
        assignments[worker] = remaining_jobs.pop()

    return assignments


def create_matrix_entries(num_workers, num_jobs):
    entries = []

    # Create labels for jobs (X-axis)
    for j in range(num_jobs):
        label_text = f"Delivery Point {j + 1}"
        label = tk.Label(matrix_frame, text=label_text)
        label.grid(row=0, column=j + 1, padx=5, pady=5, sticky=tk.NSEW)

        # Center label horizontally and vertically
        matrix_frame.grid_columnconfigure(j + 1, weight=1)
        matrix_frame.grid_rowconfigure(0, weight=1)

    for i in range(num_workers):
        # Create label for worker (Y-axis)
        label_text = f"Courier {i + 1}"
        label = tk.Label(matrix_frame, text=label_text)
        label.grid(row=i + 1, column=0, padx=5, pady=5, sticky=tk.NSEW)

        row_entries = []
        for j in range(num_jobs):
            entry = tk.Entry(matrix_frame)
            entry.grid(row=i + 1, column=j + 1, padx=5, pady=5, sticky=tk.NSEW)
            row_entries.append(entry)
        entries.append(row_entries)

    return entries


def update_matrix_entries():
    global matrix_entries
    num_workers = int(entry_workers.get())
    num_jobs = int(entry_jobs.get())

    for row in matrix_entries:
        for entry in row:
            entry.destroy()

    matrix_entries = create_matrix_entries(num_workers, num_jobs)


def generate_random_values():
    num_workers = int(entry_workers.get())
    num_jobs = int(entry_jobs.get())

    for i in range(num_workers):
        for j in range(num_jobs):
            entry = matrix_entries[i][j]
            value = np.random.randint(1, 11)  # Generate random value between 1 and 10
            entry.delete(0, tk.END)
            entry.insert(tk.END, str(value))


def solve_maximization():
    global cost_matrix  # Access the global cost_matrix variable
    num_workers = int(entry_workers.get())
    num_jobs = int(entry_jobs.get())

    matrix = []
    for i in range(num_workers):
        row = []
        for j in range(num_jobs):
            entry = matrix_entries[i][j]
            value = entry.get()
            if not value.isdigit():  # Перевірка чи введене значення є цифрою
                tk.messagebox.showwarning("Invalid Input", "Please, use only numbers for data input.")
                return
            row.append(float(value))
        matrix.append(row)

    cost_matrix = np.array(matrix)
    assignment = hungarian_algorithm(cost_matrix, maximize=True)

    sorted_assignments = sorted(assignment.items(), key=lambda x: cost_matrix[x[0], x[1]], reverse=True)

    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, "Assignments (Maximization):\n")

    for worker, job in sorted_assignments:
        result_text.insert(tk.END, f"Courier {worker + 1} assigned to Delivery Point {job + 1}\n")

    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "Output Matrix (Maximization):\n")
    output_matrix = np.zeros_like(cost_matrix, dtype=int)
    for worker, job in assignment.items():
        output_matrix[worker, job] = 1
    output_text.insert(tk.END, np.array2string(output_matrix, separator="\t"))


def solve_minimization():
    global cost_matrix  # Access the global cost_matrix variable
    num_workers = int(entry_workers.get())
    num_jobs = int(entry_jobs.get())

    matrix = []
    for i in range(num_workers):
        row = []
        for j in range(num_jobs):
            entry = matrix_entries[i][j]
            value = entry.get()
            if not value.isdigit():  # Перевірка чи введене значення є цифрою
                tkinter.messagebox.showwarning("Invalid Input", "Please, use only numbers for data input.")
                return
            row.append(float(value))
        matrix.append(row)

    cost_matrix = np.array(matrix)
    assignment = hungarian_algorithm(cost_matrix, maximize=False)

    sorted_assignments = sorted(assignment.items(), key=lambda x: cost_matrix[x[0], x[1]])

    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, "Assignments (Minimization):\n")

    for worker, job in sorted_assignments:
        result_text.insert(tk.END, f"Courier {worker + 1} assigned to Delivery Point {job + 1}\n")

    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "Output Matrix (Minimization):\n")
    output_matrix = np.zeros_like(cost_matrix, dtype=int)
    for worker, job in assignment.items():
        output_matrix[worker, job] = 1
    output_text.insert(tk.END, np.array2string(output_matrix, separator="\t"))


def show_cell_efficiency():
    if cost_matrix is None:
        return

    num_workers, num_jobs = cost_matrix.shape

    fig, ax = plt.subplots()
    ax.imshow(cost_matrix, cmap='Blues')
    ax.set_xticks(np.arange(num_jobs))
    ax.set_yticks(np.arange(num_workers))
    ax.set_xticklabels(np.arange(1, num_jobs + 1))
    ax.set_yticklabels(np.arange(1, num_workers + 1))
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    for i in range(num_workers):
        for j in range(num_jobs):
            value = cost_matrix[i, j]
            text = ax.text(j, i, value, ha='center', va='center', color='black')

    plt.show()

def show_full_graph():
    if cost_matrix is None:
        return

    num_workers, num_jobs = cost_matrix.shape

    # Create a bipartite graph
    G = nx.Graph()

    workers = [f"Courier {i+1}" for i in range(num_workers)]
    jobs = [f"Point {j+1}" for j in range(num_jobs)]

    G.add_nodes_from(workers, bipartite=0)
    G.add_nodes_from(jobs, bipartite=1)

    for worker, job in assignments.items():
        G.add_edge(f"Courier {worker+1}", f"Point {job+1}")

    # Create a complete graph with all connections
    for worker in workers:
        for job in jobs:
            if not G.has_edge(worker, job):
                G.add_edge(worker, job)

    # Calculate the efficiency criterion for each connection
    efficiency = {}
    for worker in workers:
        for job in jobs:
            if G.has_edge(worker, job):
                worker_id = int(worker.split()[1])
                job_id = int(job.split()[1])
                efficiency[(worker, job)] = cost_matrix[worker_id-1, job_id-1]

    # Sort the connections by efficiency criterion
    sorted_connections = sorted(efficiency.items(), key=lambda x: x[1])

    # Determine if it is maximization or minimization
    maximize = True if len(assignments) > 0 else False

    # Determine if the checkbox is checked
    hide_zero_efficiency = checkbox_var.get()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Compute the spring layout for improved edge routing
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, nodelist=workers, node_color='skyblue', node_size=500, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=jobs, node_color='lightpink', node_size=500, ax=ax)

    nx.draw_networkx_labels(G, pos, font_color='black', font_size=10, ax=ax)

    nx.draw_networkx_edges(G, pos, edgelist=[e for e, v in sorted_connections if v != 0.0 or not hide_zero_efficiency], edge_color='gray', width=1.0, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={e: v for e, v in sorted_connections if v != 0.0 or not hide_zero_efficiency}, font_color='black', font_size=8, ax=ax)

    ax.set_title("Bipartite Graph with Efficiency Criterion")
    plt.show()


window = tk.Tk()
window.title("Transportation Problem")
window.geometry("800x600")

window.grid_rowconfigure(1, weight=1)
window.grid_columnconfigure(0, weight=1)

main_frame = ttk.Frame(window)
main_frame.pack(fill="both", expand=True)

label_workers = ttk.Label(main_frame, text="Number of Couriers:")
label_workers.pack(pady=(10, 5))

entry_workers = ttk.Entry(main_frame)
entry_workers.pack()

label_jobs = ttk.Label(main_frame, text="Number of delivery locations:")
label_jobs.pack(pady=(10, 5))

entry_jobs = ttk.Entry(main_frame)
entry_jobs.pack()

matrix_frame = ttk.Frame(main_frame)
matrix_frame.pack(pady=10)

button_update = ttk.Button(main_frame, text="Update", command=update_matrix_entries)
button_update.pack(pady=5)

button_generate = ttk.Button(main_frame, text="Generate Random Values", command=generate_random_values)
button_generate.pack(pady=5)

separator1 = ttk.Separator(main_frame, orient="horizontal")
separator1.pack(fill="x", pady=10)

result_frame = ttk.Frame(main_frame)
result_frame.pack(fill="both", expand=True)

result_label = ttk.Label(result_frame, text="Assignments:")
result_label.pack(pady=(10, 5))

result_text = tk.Text(result_frame, height=6, width=40)
result_text.pack(fill="both", expand=True)

separator2 = ttk.Separator(main_frame, orient="horizontal")
separator2.pack(fill="x", pady=10)

output_frame = ttk.Frame(main_frame)
output_frame.pack(fill="both", expand=True)

output_label = ttk.Label(output_frame, text="Output Matrix:")
output_label.pack(pady=(10, 5))

output_text = tk.Text(output_frame, height=6, width=40)
output_text.pack(fill="both", expand=True)

button_solve_maximization = ttk.Button(main_frame, text="Solve (Maximization)", command=solve_maximization)
button_solve_maximization.pack(pady=10, side=tk.LEFT)

button_solve_minimization = ttk.Button(main_frame, text="Solve (Minimization)", command=solve_minimization)
button_solve_minimization.pack(side=tk.LEFT)

button_show_efficiency = ttk.Button(main_frame, text="Cell Efficiency", command=show_cell_efficiency)
button_show_efficiency.pack(side=tk.LEFT, padx=10, pady=10)

checkbox_var = tk.BooleanVar()
checkbox = ttk.Checkbutton(main_frame, text="Hide Zero Efficiency", variable=checkbox_var)
checkbox.pack(side=tk.LEFT, padx=10, pady=10)

button_full_graph = ttk.Button(main_frame, text="Show Full Graph", command=show_full_graph)
button_full_graph.pack(side=tk.RIGHT, padx=10, pady=5)

window.mainloop()

# works fine - stable version