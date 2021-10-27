from mocafe.fenut.fenut import flatten_list_of_lists


def test_flatten_list_of_lists_partial_empty():
    list_of_lists = [[1, 2], [], [], []]
    list = flatten_list_of_lists(list_of_lists)
    list_ref = [1, 2]
    assert list == list_ref, "They should be equal"


def test_flatten_list_of_lists_partial_empty2():
    list_of_lists = [[1], [], [], []]
    list = flatten_list_of_lists(list_of_lists)
    list_ref = [1]
    assert list == list_ref, "They should be equal"


def test_flatten_list_of_lists_empty():
    list_of_lists = [[], [], [], []]
    list = flatten_list_of_lists(list_of_lists)
    list_ref = []
    assert list == list_ref, "They should be equal"
