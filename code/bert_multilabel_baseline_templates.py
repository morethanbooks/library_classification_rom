
templates = {
    'eng': {
        # start
        'the book': 'The book', 
        'written': 'written',
        # parameterized
        'by': (lambda first, last: 'by' + (' ' + first if first else '') + (' ' + last if last else ''), 
            ('author_first_name', 'author_last_name')),
        'in': ('in', ('year',)),
        'published in': ('published in', ('place_publication',)),
        'with a length of': ('with a length of {} pages'.format, ('pages',)),
        'edited by': (
            lambda first, last: 'edited by' + (' ' + first if first else '') + (' ' + last if last else ''),
            ('editor_first_name', 'editor_last_name')),
    },
    'spa': {
        # start
        'the book': 'El libro', 
        'written': 'escrito',
        # parameterized
        'by': (lambda first, last: 'por' + (' ' + first if first else '') + (' ' + last if last else ''), 
            ('author_first_name', 'author_last_name')),
        'in': ('en', ('year',)),
        'published in': ('publicado en', ('place_publication',)),
        'with a length of': ('con una extensión de {} páginas'.format, ('pages',)),
        'edited by': (
            lambda first, last: 'editado por' + (' ' + first if first else '') + (' ' + last if last else ''),
            ('editor_first_name', 'editor_last_name')),
    },
    'fre': {
        # start
        'the book': 'Le livre', 
        'written': 'écrit',
        # parameterized
        'by': (lambda first, last: 'par' + (' ' + first if first else '') + (' ' + last if last else ''), 
            ('author_first_name', 'author_last_name')),
        'in': ('en', ('year',)),
        'published in': ('publié à', ('place_publication',)),
        'with a length of': ("d'une longueur de {} pages".format, ('pages',)),
        'edited by': (
            lambda first, last: 'édité par' + (' ' + first if first else '') + (' ' + last if last else ''),
            ('editor_first_name', 'editor_last_name')),
    },
    'ger': {
        # start
        'the book': 'Das Buch', 
        'written': 'geschrieben',
        # parameterized
        'by': (lambda first, last: 'von' + (' ' + first if first else '') + (' ' + last if last else ''), 
            ('author_first_name', 'author_last_name')),
        'in': ('in', ('year',)),
        'published in': ('veröffentlicht in', ('place_publication',)),
        'with a length of': ('mit einer Länge von {} Seiten'.format, ('pages',)),
        'edited by': (
            lambda first, last: 'herausgegeben durch' + (' ' + first if first else '') + (' ' + last if last else ''),
            ('editor_first_name', 'editor_last_name')),
    },
    'ita': {
        # start
        'the book': 'Il libro', 
        'written': 'scritto',
        # parameterized
        'by': (lambda first, last: 'di' + (' ' + first if first else '') + (' ' + last if last else ''), 
            ('author_first_name', 'author_last_name')),
        'in': ('nel', ('year',)),
        'published in': ('pubblicato a', ('place_publication',)),
        'with a length of': ('con una lunghezza di {} pagine'.format, ('pages',)),
        'edited by': (
            lambda first, last: 'a cura di' + (' ' + first if first else '') + (' ' + last if last else ''),
            ('editor_first_name', 'editor_last_name')),
    },
    'por': {
        # start
        'the book': 'O livro', 
        'written': 'escrito',
        # parameterized
        'by': (lambda first, last: 'por' + (' ' + first if first else '') + (' ' + last if last else ''), 
            ('author_first_name', 'author_last_name')),
        'in': ('em', ('year',)),
        'published in': ('publicado em', ('place_publication',)),
        'with a length of': ('com una extensão de {} páginas'.format, ('pages',)),
        'edited by': (
            lambda first, last: 'editado por' + (' ' + first if first else '') + (' ' + last if last else ''),
            ('editor_first_name', 'editor_last_name')),
    },
    'cat': {
        # start
        'the book': 'El llibre', 
        'written': 'escrit',
        # parameterized
        'by': (lambda first, last: 'per' + (' ' + first if first else '') + (' ' + last if last else ''), 
            ('author_first_name', 'author_last_name')),
        'in': ("l'any", ('year',)),
        'published in': ('publicat a', ('place_publication',)),
        'with a length of': ('amb una extensió de {} pàgines'.format, ('pages',)),
        'edited by': (
            lambda first, last: 'editat per' + (' ' + first if first else '') + (' ' + last if last else ''),
            ('editor_first_name', 'editor_last_name')),
    },
    'rum': {
        # start
        'the book': 'Cartea', 
        'written': 'scrisă',
        # parameterized
        'by': (lambda first, last: 'de' + (' ' + first if first else '') + (' ' + last if last else ''), 
            ('author_first_name', 'author_last_name')),
        'in': ("în", ('year',)),
        'published in': ('publicată la', ('place_publication',)),
        'with a length of': ('cu o lungime de {} pagini'.format, ('pages',)),
        'edited by': (
            lambda first, last: 'editată de' + (' ' + first if first else '') + (' ' + last if last else ''),
            ('editor_first_name', 'editor_last_name')),
    },
    'glg': {
        # start
        'the book': 'O libro', 
        'written': 'escrito',
        # parameterized
        'by': (lambda first, last: 'por' + (' ' + first if first else '') + (' ' + last if last else ''), 
            ('author_first_name', 'author_last_name')),
        'in': ("en", ('year',)),
        'published in': ('publicado en', ('place_publication',)),
        'with a length of': ('cunha extensión de {} paxinas'.format, ('pages',)),
        'edited by': (
            lambda first, last: 'editado por' + (' ' + first if first else '') + (' ' + last if last else ''),
            ('editor_first_name', 'editor_last_name')),
    },
}
