import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def read_coords_from_txt(filename):
    coords = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                try:
                    x, y = map(int, line.strip().split(','))
                    coords.append((x, y))
                except:
                    continue
    return coords

def save_coords_to_txt(filename, coords):
    with open(filename, 'w') as f:
        for x, y in coords:
            f.write(f"{x},{y}\n")
    print(f"Saved {len(coords)} coords to {filename}")

def interactive_image_marker(image_path, coord_file):
    img = mpimg.imread(image_path)
    h, w = img.shape[:2]

    initial_coords = read_coords_from_txt(coord_file)

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img, interpolation='none')
    ax.axis('off')

    marked_points = {}
    point_stack = []
    coords_list = []
    undo_stack = []

    # 初始化已有坐标
    i = 1
    for x, y in initial_coords:
        dot = ax.plot(x, y, 'ro', markersize=4)[0]
        text = ax.text(x + 5, y, i, color='red', fontsize=12)
        marked_points[(x, y)] = (dot, text)
        point_stack.append((x, y))
        coords_list.append((x, y))
        i += 1

    print(f"Loaded {len(initial_coords)} coords from {coord_file}")

    def onclick(event):
        if event.inaxes and event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)
            key = (x, y)

            if event.button == 1:  # 左键添加
                if key not in marked_points:
                    number = len(marked_points) + 1
                    dot = ax.plot(x, y, 'ro', markersize=4)[0]
                    text = ax.text(x + 5, y, f'{number}', color='red', fontsize=12)
                    marked_points[key] = (dot, text)
                    point_stack.append(key)
                    coords_list.append(key)
                    undo_stack.clear()
                    print(f"Marked {number} at ({x}, {y})")
                else:
                    print(f"Already marked: ({x}, {y})")

            elif event.button == 3:  # 右键点击
                if event.key == 'shift':  # Shift + 右键 => 恢复
                    if undo_stack:
                        key = undo_stack.pop()
                        number = len(marked_points) + 1
                        dot = ax.plot(key[0], key[1], 'ro', markersize=4)[0]
                        text = ax.text(key[0] + 5, key[1], f'{number}', color='red', fontsize=12)
                        marked_points[key] = (dot, text)
                        point_stack.append(key)
                        coords_list.append(key)
                        print(f"Restored: {key}")
                    else:
                        print("Nothing to restore.")
                else:  # 普通右键 => 撤销
                    if point_stack:
                        last_key = point_stack.pop()
                        dot, text = marked_points[last_key]
                        dot.remove()
                        text.remove()
                        del marked_points[last_key]
                        coords_list.remove(last_key)
                        undo_stack.append(last_key)
                        print(f"Removed: {last_key}")
                    else:
                        print("No points to remove.")

            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)

    def on_close(event):
        save_coords_to_txt(coord_file, coords_list)

    fig.canvas.mpl_connect('close_event', on_close)
    plt.show()

# interactive_image_marker("milestone1\Data\IMG_1223.JPG", coord_file='milestone1\Data\A.txt')
# interactive_image_marker("milestone1\Data\IMG_1224.JPG", coord_file='milestone1\Data\B.txt')
interactive_image_marker("milestone1\Data\IMG_1226.JPG", coord_file='milestone1\Data\C.txt')