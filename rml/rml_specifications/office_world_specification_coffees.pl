:- module('spec', [trace_expression/2, match/2]).
:- use_module(monitor('deep_subdict')).
match(_event, coffee_pick_up) :- deep_subdict(_{'f':T}, _event), T=1.0.
match(_event, mail_pick_up(N)) :- deep_subdict(_{'e':N}, _event), >(N, 0).
match(_event, drop_off) :- deep_subdict(_{'g':T}, _event), T=1.0.
match(_event, not_efg) :- not(match(_event, mail_pick_up(N))), not(match(_event, coffee_pick_up)), not(match(_event, drop_off)).
match(_, any).
trace_expression('Main', Main) :- Main=(star((not_efg:eps))*var(n, ((mail_pick_up(var(n)):eps)*app(Coffee, [var('n'), var('n')])))), Coffee=gen(['s', 'm'], guarded((var('s')>0), (star((not_efg:eps))*((coffee_pick_up:eps)*app(Coffee, [(var('s')-1), var('m')]))), app(Drop, [var('m')]))), Drop=gen(['s'], guarded((var('s')>0), (star((not_efg:eps))*((drop_off:eps)*app(Drop, [(var('s')-1)]))), star((any:eps)))).
