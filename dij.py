INF = 10 ** 20


def dijkstra(start, graph, visited=None, ways=None):
    if visited is None:
        visited = set()
        ways = [INF for x in range(len(graph))]
        ways[start] = 0
    visited.add(start)
    for vertex in range(len(graph)):
        if vertex not in visited and vertex != start \
                and graph[start][vertex] != INF:
            ways[vertex] = min(ways[vertex],
                               graph[start][vertex] + ways[start])
    if len(visited) < len(graph):
        min_way = INF + 1
        next_vertex = -1
        for vertex in range(len(graph)):
            if vertex not in visited:
                if ways[vertex] < min_way:
                    min_way = ways[vertex]
                    next_vertex = vertex
        return dijkstra(next_vertex, graph, visited, ways)
    else:
        return ways


def make_graph(vertex_num):
    graph = []
    for i in range(vertex_num):
        string = map(lambda x: INF if x < 0 else x, map(int, input().split()))
        graph.append(list(string))
    return graph


def main():
    vertex_num, start_vertex, end_vertex = map(int, input().split())
    graph = make_graph(vertex_num)
    way = dijkstra(start_vertex - 1, graph)[end_vertex - 1]
    if way != INF:
        print(way)
    else:
        print(-1)


if __name__ == '__main__':
    main()

