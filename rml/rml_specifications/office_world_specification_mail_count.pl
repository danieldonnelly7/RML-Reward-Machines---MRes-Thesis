:- module('spec', [trace_expression/2, match/2]).
:- use_module(monitor('deep_subdict')).
match(_event, coffee_pick_up) :- deep_subdict(_{'f':T}, _event), T=1.0.
match(_event, mail_pick_up) :- deep_subdict(_{'e':T}, _event), T=1.0.
match(_event, out_of_mail) :- deep_subdict(_{'o':T}, _event), T=1.0.
match(_event, drop_off) :- deep_subdict(_{'g':T}, _event), T=1.0.
match(_event, any_n) :- deep_subdict(_{'nn':T}, _event), T=1.0.
match(_event, not_efgon) :- not(match(_event, mail_pick_up)), not(match(_event, coffee_pick_up)), not(match(_event, drop_off)), not(match(_event, out_of_mail)), not(match(_event, any_n)).
match(_, any).
trace_expression('Main', Main) :- Main=(star((not_efgon:eps))*app(Mail, [1])), Mail=gen(['n'], ((mail_pick_up:eps)*(star((not_efgon:eps))*(app(Mail, [(var('n')+1)])\/((out_of_mail:eps)*app(Coffee, [var('n'), var('n')])))))), Coffee=gen(['s', 'm'], guarded((var('s')>0), (star((not_efgon:eps))*((coffee_pick_up:eps)*app(Coffee, [(var('s')-1), var('m')]))), app(Drop, [var('m')]))), Drop=gen(['s'], guarded((var('s')>0), (star((not_efgon:eps))*((drop_off:eps)*app(Drop, [(var('s')-1)]))), 1)).
