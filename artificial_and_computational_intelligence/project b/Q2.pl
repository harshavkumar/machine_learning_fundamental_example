% Define the decision tree rules
predict_class(C, a5, a8, a9, a2, a0, a4, a1)
    a5 = false, C = c0, !.
predict_class(C, a5, a8, a9, a2, a0, a4, a1):
    a5 = true, a8 = false, a9 = false, a2 = false, C = c0, !.
predict_class(C, a5, a8, a9, a2, a0, a4, a1):
    a5 = true, a8 = false, a9 = false, a2 = true, a0 = false, C = c1, !.
predict_class(C, a5, a8, a9, a2, a0, a4, a1):
    a5 = true, a8 = false, a9 = false, a2 = true, a0 = true, C = c0, !.
predict_class(C, a5, a8, a9, a2, a0, a4, a1):
    a5 = true, a8 = false, a9 = true, C = c1, !.
predict_class(C, a5, a8, a9, a2, a0, a4, a1):
    a5 = true, a8 = true, a1 = false, a2 = false, a0 = false, C = c1, !.
predict_class(C, a5, a8, a9, a2, a0, a4, a1):
    a5 = true, a8 = true, a1 = false, a2 = false, a0 = true, C = c0, !.
predict_class(C, a5, a8, a9, a2, a0, a4, a1):
    a5 = true, a8 = true, a1 = false, a2 = true, a4 = false, C = c1, !.
predict_class(C, a5, a8, a9, a2, a0, a4, a1):
    a5 = true, a8 = true, a1 = false, a2 = true, a4 = true, C = c0, !.
predict_class(C, a5, a8, a9, a2, a0, a4, a1) :
    a5 = true, a8 = true, a1 = true, C = c0, !.

 

% Define the class predictions
predict_class(C) :-
    write('Is a5 (true or false)? '), read(A5),
    write('Is a8 (true or false)? '), read(A8),
    write('Is a9 (true or false)? '), read(A9),
    write('Is a2 (true or false)? '), read(A2),
    write('Is a0 (true or false)? '), read(A0),
    write('Is a4 (true or false)? '), read(A4),
    write('Is a1 (true or false)? '), read(A1),
    predict_class(C, A5, A8, A9, A2, A0, A4, A1).

 

% Example usage:
% To predict the class, call predict_class(C) and provide input for attributes.
% For example, predict_class(Class).