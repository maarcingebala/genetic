# coding: utf-8

import random
import collections
from clint.textui import puts
from clint.textui.cols import columns


CHROMOSOME_LONG_REPR = True

MAX_FITNESS = 1000
WEEK_DAYS = {0:'PONIEDZIAŁEK', 1:'WTOREK', 2:'ŚRODA', 3:'CZWARTEK', 4:'PIĄTEK', 5:'SOBOTA', 6:'NIEDZIELA'}


def _int_to_bin(x, fill):
    if not x >= 0:
        raise TypeError('Unsupported value')
    return bin(x)[2:].zfill(fill)


class Chromosome(object):

    obj_counter = 0

    def __init__(self, groups_num=None, rooms_num=None, work_hours_num=6, days_num=1, genotype=''):
        self.groups_num = groups_num
        self.rooms_num = rooms_num
        self.work_hours_num = work_hours_num
        self.days_num = days_num
        self.genotype = '' or genotype

        if not self.genotype:
            if not groups_num or not rooms_num:
                raise TypeError('groups_num and rooms_num not specified')
            self.generate()
        
        Chromosome.obj_counter += 1
        self.no = Chromosome.obj_counter
        self.special = False

    def generate(self):
        genotype = ''
        group_code_length = len(bin(self.groups_num)[2:])
        for day in range(self.days_num):
            for work_hour in range(self.work_hours_num):
                for room in range(self.rooms_num):
                    gr = random.randint(0, self.groups_num)
                    genotype += _int_to_bin(gr, group_code_length)
        
        # if not self.is_correct():
        #     raise TypeError('Invalid chromosome: %s' % genotype)
        
        self.genotype = genotype

    def decode(self):
        decoded = {}
        day_length = len(self.genotype) / self.days_num
        single_hour_length = day_length / self.work_hours_num
        group_code_length = len(bin(self.groups_num)[2:])
        for day in range(self.days_num):
            decoded[day] = {}
            for work_hour in range(self.work_hours_num):
                decoded[day][work_hour] = {}
                for room in range(self.rooms_num):
                    start = day * day_length + work_hour * single_hour_length + room * group_code_length
                    end = start + group_code_length
                    group = int(self.genotype[start:end], 2)
                    decoded[day][work_hour][room] = str(group) if group != 0 else '-'
        return decoded

    def is_correct(self):
        allowed = [str(gr) for gr in range(1, self.groups_num + 1)]
        allowed.append('-')
        for day, schedule in self.decode().iteritems():
            for hour, room_group in schedule.iteritems():
                diff = set(room_group.values()) - set(allowed)
                if diff:
                    return False
        return True

    def collisions(self):
        total = 0
        for day, schedule in self.decode().iteritems():
            for hour, room_group in schedule.iteritems():
                c = collections.Counter(room_group.values())
                for room, num in c.iteritems():
                    if num > 1 and room not in ['0', '-']:
                        total += num - 1
        return total

    def gaps(self):
        total = 0
        groups = [str(gr) for gr in range(1, self.groups_num + 1)]
        for day, schedule in self.decode().iteritems():
            for gr in groups:
                _gaps = []
                for hour, room_group in schedule.iteritems():
                    if gr in room_group.values():
                        _gaps.append(str(hour))
                    else:
                        _gaps.append('x')
                total += ''.join(_gaps).strip('x').count('x')
        return total

    def fitness(self):
        return MAX_FITNESS - self.collisions() - self.gaps()

    def get_properties(self):
        return {'groups_num': self.groups_num,
            'rooms_num': self.rooms_num,
            'work_hours_num': self.work_hours_num,
            'days_num': self.days_num}

    def show_decoded(self):
        def _str_gr(gr):
            if not gr == '-':
                gr = 'gr%s' % gr
            return gr

        col_width = 8
        data = self.decode()
        for day, schedule in data.iteritems():
            print '\n', WEEK_DAYS[day]
            print len(WEEK_DAYS[day]) * '='
            header = [['sala %s' % room, col_width] for room in schedule[0].keys()]
            puts(columns(*header))
            for room_group in schedule.itervalues():
                row = [[_str_gr(gr), col_width] for gr in room_group.itervalues()]
                puts(columns(*row))

    def __str__(self):
        _str = 'ch%s' % self.no
        if CHROMOSOME_LONG_REPR:
            _str += ': <%s> (%s, %s)' % (self.genotype, self.collisions(), self.gaps())
        if self.special:
            _str += '*'
        return _str


class Population(list):

    def total_collisions(self):
        return sum([ch.collisions() for ch in self.__iter__()])

    def best_ch(self):
        return max(self.__iter__(), key=lambda ch: ch.fitness())

    def pop_best(self):
        b = self.best_ch()
        return self.pop(self.index(b))

    def worst_ch(self):
        return min(self.__iter__(), key=lambda ch: ch.fitness())

    def best_f(self):
        return self.best_ch().fitness()

    def worst_f(self):
        return self.worst_ch().fitness()

    def avg_f(self):
        return sum([ch.fitness() for ch in self.__iter__()]) / float(self.__len__())

    def __str__(self):
        _str = '[ '
        for ch in self.__iter__():
            if CHROMOSOME_LONG_REPR:
                _str += '\n'
            _str += '%s ' % ch
        _str += ']'
        return _str

