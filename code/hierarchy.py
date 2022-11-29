
import collections
import pandas as pd


def bfs(graph, start='root'):
    queue, visited = [], set()
    level, parent = 0, None
    queue.append((start, parent, level))

    while queue:
        s, parent, level = queue.pop(0)
        yield s, parent, level

        # check if terminal node
        for node in graph.get(s, []):
            if node not in visited:
                visited.add(node)
                queue.append((node, s, level + 1))


def dfs(graph, start='root'):
    def _dfs(visited, graph, node, level):
        if node not in visited:
            yield node, level
            visited.add(node)
            for child in graph.get(node, []):
                yield from _dfs(visited, graph, child, level + 1)

    visited = set()
    yield from _dfs(visited, graph, start, 0)


def map_to_max_level(tree, max_level):
    node_to_parent = {}
    node_levels = {}
    for _, row in tree.iterrows():
        node_to_parent[row['node']] = row['parent']
        node_levels[row['node']] = row['level']

    mapping = {}
    for _, row in tree.iterrows():
        if node_levels[row['node']] > max_level:
            parent = node_to_parent[row['node']]
            while node_levels[parent] > max_level:
                parent = node_to_parent[parent]
            mapping[row['node']] = parent
        else:
            mapping[row['node']] = row['node']

    return mapping


if __name__ == '__main__':
    tree = pd.read_csv('data/bk/BK.tsv', sep='\t')

    ppn2bk = {}
    for _, row in tree.iterrows():
        ppn2bk[row['ppn']] = row['notation']

    parents = {}
    for _, row in tree.iterrows():
        parents[row['notation']] = ppn2bk.get(row['nueb'], None)

    # export
    graph = collections.defaultdict(list)
    for _, row in tree.iterrows():
        parent = ppn2bk.get(row['nueb'], 'root')
        graph[parent].append(row['notation'])
    with open('./data/label-hierarchy.tsv', 'w+') as f:
        f.write('\t'.join(['node', 'parent', 'level']) + '\n')
        data = bfs(graph)
        # skip dummy root
        next(data)
        for node, parent, level in data:
            f.write('\t'.join([node, parent, str(level)]) + '\n')
