import copy
import numpy
from inspect import getfullargspec as getargspec
import collections
import numbers
import itertools
import warnings
import sys
import re
import math
from math import sqrt, log, isnan, isinf


def isinfinite(x):
    return isinf(x) or isnan(x)


FLOAT_LIKE_TYPES = (numbers.Number, numpy.generic)
CONSTANT_TYPES = FLOAT_LIKE_TYPES + (complex,)


def correlated_values(nom_values, covariance_mat, tags=None):
    std_devs = numpy.sqrt(numpy.diag(covariance_mat))
    norm_vector = std_devs.copy()
    norm_vector[norm_vector == 0] = 1

    return correlated_values_norm(
        list(zip(nom_values, std_devs)),
        covariance_mat/norm_vector/norm_vector[:, numpy.newaxis],
        tags)


def correlated_values_norm(values_with_std_dev, correlation_mat, tags=None):
    if tags is None:
        tags = (None,) * len(values_with_std_dev)

    (nominal_values, std_devs) = numpy.transpose(values_with_std_dev)

    (variances, transform) = numpy.linalg.eigh(correlation_mat)

    variances[variances < 0] = 0.

    variables = tuple(
        Variable(0, sqrt(variance), tag)
        for (variance, tag) in zip(variances, tags))

    transform *= std_devs[:, numpy.newaxis]

    values_funcs = tuple(
        AffineScalarFunc(
            value,
            LinearCombination(dict(zip(variables, coords))))
        for (coords, value) in zip(transform, nominal_values))

    return values_funcs


class NotUpcast(Exception):
    'Raised when an object cannot be converted to a number with uncertainty'


def to_affine_scalar(x):
    if isinstance(x, AffineScalarFunc):
        return x

    if isinstance(x, CONSTANT_TYPES):
        return AffineScalarFunc(x, LinearCombination({}))

    raise NotUpcast("%s cannot be converted to a number with"
                    " uncertainty" % type(x))


STEP_SIZE = sqrt(sys.float_info.epsilon)


def partial_derivative(f, arg_ref):
    change_kwargs = isinstance(arg_ref, str)

    def partial_derivative_of_f(*args, **kwargs):
        if change_kwargs:
            args_with_var = kwargs
        else:
            args_with_var = list(args)

        step = STEP_SIZE*abs(args_with_var[arg_ref])
        if not step:
            step = STEP_SIZE

        args_with_var[arg_ref] += step

        if change_kwargs:
            shifted_f_plus = f(*args, **args_with_var)
        else:
            shifted_f_plus = f(*args_with_var, **kwargs)

        args_with_var[arg_ref] -= 2*step

        if change_kwargs:
            shifted_f_minus = f(*args, **args_with_var)
        else:
            shifted_f_minus = f(*args_with_var, **kwargs)

        return (shifted_f_plus - shifted_f_minus)/2/step

    return partial_derivative_of_f


class NumericalDerivatives(object):
    def __init__(self, function):
        self._function = function

    def __getitem__(self, n):
        return partial_derivative(self._function, n)


class IndexableIter(object):
    def __init__(self, iterable, none_converter=lambda index: None):
        self.iterable = iterable
        self.returned_elements = []
        self.none_converter = none_converter

    def __getitem__(self, index):

        returned_elements = self.returned_elements

        try:

            return returned_elements[index]

        except IndexError:  # Element not yet cached

            for pos in range(len(returned_elements), index+1):

                value = next(self.iterable)

                if value is None:
                    value = self.none_converter(pos)

                returned_elements.append(value)

            return returned_elements[index]

    def __str__(self):
        return '<%s: [%s...]>' % (
            self.__class__.__name__,
            ', '.join(map(str, self.returned_elements)))


def wrap(f, derivatives_args=[], derivatives_kwargs={}):
    derivatives_args_index = IndexableIter(
        itertools.chain(derivatives_args, itertools.repeat(None)))

    derivatives_all_kwargs = {}

    for (name, derivative) in derivatives_kwargs.items():

        if derivative is None:
            derivatives_all_kwargs[name] = partial_derivative(f, name)
        else:
            derivatives_all_kwargs[name] = derivative

    try:
        argspec = getargspec(f)
    except TypeError:
        pass
    else:
        for (index, name) in enumerate(argspec.args):

            derivative = derivatives_args_index[index]

            if derivative is None:
                derivatives_all_kwargs[name] = partial_derivative(f, name)
            else:
                derivatives_all_kwargs[name] = derivative

    def none_converter(index):
        return partial_derivative(f, index)

    for (index, derivative) in enumerate(
            derivatives_args_index.returned_elements):
        if derivative is None:
            derivatives_args_index.returned_elements[index] = (
                none_converter(index))

    derivatives_args_index.none_converter = none_converter

    # Wrapped function:
    def f_with_affine_output(*args, **kwargs):
        pos_w_uncert = [index for (index, value) in enumerate(args)
                        if isinstance(value, AffineScalarFunc)]
        names_w_uncert = [key for (key, value) in kwargs.items()
                          if isinstance(value, AffineScalarFunc)]
        if (not pos_w_uncert) and (not names_w_uncert):
            return f(*args, **kwargs)
        args_values = list(args)

        for index in pos_w_uncert:
            args_values[index] = args[index].nominal_value

        kwargs_uncert_values = {}

        for name in names_w_uncert:
            value_with_uncert = kwargs[name]
            kwargs_uncert_values[name] = value_with_uncert
            kwargs[name] = value_with_uncert.nominal_value

        f_nominal_value = f(*args_values, **kwargs)

        if not isinstance(f_nominal_value, FLOAT_LIKE_TYPES):
            return NotImplemented

        linear_part = []

        for pos in pos_w_uncert:
            linear_part.append((
                derivatives_args_index[pos](*args_values, **kwargs),
                args[pos]._linear_part))

        for name in names_w_uncert:
            derivative = derivatives_all_kwargs.setdefault(
                name, partial_derivative(f, name))

            linear_part.append((
                derivative(*args_values, **kwargs),
                kwargs_uncert_values[name]._linear_part))

        return AffineScalarFunc(f_nominal_value, LinearCombination(linear_part))

    f_with_affine_output.name = f.__name__

    return f_with_affine_output


def force_aff_func_args(func):
    def op_on_upcast_args(x, y):
        try:
            y_with_uncert = to_affine_scalar(y)
        except NotUpcast:
            return NotImplemented
        else:
            return func(x, y_with_uncert)

    return op_on_upcast_args


def eq_on_aff_funcs(self, y_with_uncert):
    difference = self - y_with_uncert
    return not(difference._nominal_value or difference.std_dev)


def ne_on_aff_funcs(self, y_with_uncert):
    return not eq_on_aff_funcs(self, y_with_uncert)


def gt_on_aff_funcs(self, y_with_uncert):
    return self._nominal_value > y_with_uncert._nominal_value


def ge_on_aff_funcs(self, y_with_uncert):
    return (gt_on_aff_funcs(self, y_with_uncert)
            or eq_on_aff_funcs(self, y_with_uncert))


def lt_on_aff_funcs(self, y_with_uncert):
    return self._nominal_value < y_with_uncert._nominal_value


def le_on_aff_funcs(self, y_with_uncert):
    return (lt_on_aff_funcs(self, y_with_uncert)
            or eq_on_aff_funcs(self, y_with_uncert))


def first_digit(value):
    try:
        return int(math.floor(math.log10(abs(value))))
    except ValueError:  # Case of value == 0
        return 0


def PDG_precision(std_dev):
    exponent = first_digit(std_dev)
    if exponent >= 0:
        (exponent, factor) = (exponent-2, 1)
    else:
        (exponent, factor) = (exponent+1, 1000)
    digits = int(std_dev/10.**exponent*factor)

    if digits <= 354:
        return (2, std_dev)
    elif digits <= 949:
        return (1, std_dev)
    else:
        return (2, 10.**exponent*(1000/factor))


robust_format = format
EXP_LETTERS = {'f': 'e', 'F': 'E'}
EXP_PRINT = {
    'pretty-print': lambda common_exp: u'×10%s' % to_superscript(common_exp),
    'latex': lambda common_exp: r' \times 10^{%d}' % common_exp}

GROUP_SYMBOLS = {
    'pretty-print': ('(', ')'),
    'latex': (r'\left(', r'\right)'),
    'default': ('(', ')')  # Basic text mode
}


def robust_align(orig_str, fill_char, align_option, width):
    return format(orig_str, fill_char+align_option+width)


def format_num(nom_val_main, error_main, common_exp,
               fmt_parts, prec, main_pres_type, options):
    if 'P' in options:
        print_type = 'pretty-print'
    elif 'L' in options:
        print_type = 'latex'
    else:
        print_type = 'default'

    if common_exp is None:
        exp_str = ''
    elif print_type == 'default':
        exp_str = EXP_LETTERS[main_pres_type]+'%+03d' % common_exp
    else:
        exp_str = EXP_PRINT[print_type](common_exp)

    percent_str = ''
    if '%' in options:
        if 'L' in options:
            percent_str += ' \\'
        percent_str += '%'

    special_error = not error_main or isinfinite(error_main)

    if special_error and fmt_parts['type'] in ('', 'g', 'G'):
        fmt_suffix_n = (fmt_parts['prec'] or '')+fmt_parts['type']
    else:
        fmt_suffix_n = '.%d%s' % (prec, main_pres_type)

    if 'S' in options:  # Shorthand notation:
        if error_main == 0:
            # The error is exactly zero
            uncert_str = '0'
        elif isnan(error_main):
            uncert_str = robust_format(error_main, main_pres_type)
            if 'L' in options:
                uncert_str = r'\mathrm{%s}' % uncert_str
        elif isinf(error_main):
            if 'L' in options:
                uncert_str = r'\infty'
            else:
                uncert_str = robust_format(error_main, main_pres_type)
        else:  # Error with a meaningful first digit (not 0, and real number)

            uncert = round(error_main, prec)

            if first_digit(uncert) >= 0 and prec > 0:
                uncert_str = '%.*f' % (prec, uncert)

            else:
                if uncert:
                    uncert_str = '%d' % round(uncert*10.**prec)
                else:
                    uncert_str = '0.'
        value_end = '(%s)%s%s' % (uncert_str, exp_str, percent_str)
        any_exp_factored = True  # Single exponent in the output

        if fmt_parts['zero'] and fmt_parts['width']:

            nom_val_width = max(int(fmt_parts['width']) - len(value_end), 0)
            fmt_prefix_n = '%s%s%d%s' % (
                fmt_parts['sign'], fmt_parts['zero'], nom_val_width,
                fmt_parts['comma'])

        else:
            fmt_prefix_n = fmt_parts['sign']+fmt_parts['comma']
        nom_val_str = robust_format(nom_val_main, fmt_prefix_n+fmt_suffix_n)
        if 'L' in options:

            if isnan(nom_val_main):
                nom_val_str = r'\mathrm{%s}' % nom_val_str
            elif isinf(nom_val_main):
                nom_val_str = r'%s\infty' % ('-' if nom_val_main < 0 else '')

        value_str = nom_val_str+value_end
        if fmt_parts['width']:
            value_str = robust_align(
                value_str, fmt_parts['fill'], fmt_parts['align'] or '>',
                fmt_parts['width'])

    else:  # +/- notation:
        any_exp_factored = not fmt_parts['width']
        error_has_exp = not any_exp_factored and not special_error
        nom_has_exp = not any_exp_factored and not isinfinite(nom_val_main)

        if fmt_parts['width']:  # Individual widths
            if fmt_parts['zero']:

                width = int(fmt_parts['width'])
                remaining_width = max(width-len(exp_str), 0)

                fmt_prefix_n = '%s%s%d%s' % (
                    fmt_parts['sign'], fmt_parts['zero'],
                    remaining_width if nom_has_exp else width,
                    fmt_parts['comma'])

                fmt_prefix_e = '%s%d%s' % (
                    fmt_parts['zero'],
                    remaining_width if error_has_exp else width,
                    fmt_parts['comma'])

            else:
                fmt_prefix_n = fmt_parts['sign']+fmt_parts['comma']
                fmt_prefix_e = fmt_parts['comma']

        else:  # Global width
            fmt_prefix_n = fmt_parts['sign']+fmt_parts['comma']
            fmt_prefix_e = fmt_parts['comma']

        nom_val_str = robust_format(nom_val_main, fmt_prefix_n+fmt_suffix_n)

        if error_main:
            if (isinfinite(nom_val_main)
                # Only some formats have a nicer representation:
                    and fmt_parts['type'] in ('', 'g', 'G')):
                # The error can be formatted independently:
                fmt_suffix_e = (fmt_parts['prec'] or '')+fmt_parts['type']
            else:
                fmt_suffix_e = '.%d%s' % (prec, main_pres_type)
        else:
            fmt_suffix_e = '.0%s' % main_pres_type

        error_str = robust_format(error_main, fmt_prefix_e+fmt_suffix_e)

        if 'L' in options:

            if isnan(nom_val_main):
                nom_val_str = r'\mathrm{%s}' % nom_val_str
            elif isinf(nom_val_main):
                nom_val_str = r'%s\infty' % ('-' if nom_val_main < 0 else '')

            if isnan(error_main):
                error_str = r'\mathrm{%s}' % error_str
            elif isinf(error_main):
                error_str = r'\infty'

        if nom_has_exp:
            nom_val_str += exp_str
        if error_has_exp:
            error_str += exp_str
        if fmt_parts['width']:  # An individual alignment is needed:
            effective_align = fmt_parts['align'] or '>'

            nom_val_str = robust_align(
                nom_val_str, fmt_parts['fill'], effective_align,
                fmt_parts['width'])

            error_str = robust_align(
                error_str, fmt_parts['fill'], effective_align,
                fmt_parts['width'])

        if 'P' in options:
            pm_symbol = u'±'
        elif 'L' in options:
            pm_symbol = r' \pm '
        else:
            pm_symbol = '+/-'
        (LEFT_GROUPING, RIGHT_GROUPING) = GROUP_SYMBOLS[print_type]
        if any_exp_factored and common_exp is not None:  # Exponent
            value_str = ''.join((
                LEFT_GROUPING,
                nom_val_str, pm_symbol, error_str,
                RIGHT_GROUPING,
                exp_str, percent_str))
        else:  # No exponent
            value_str = ''.join([nom_val_str, pm_symbol, error_str])
            if percent_str:
                value_str = ''.join((
                    LEFT_GROUPING, value_str, RIGHT_GROUPING, percent_str))
            elif 'p' in options:
                value_str = ''.join((LEFT_GROUPING, value_str, RIGHT_GROUPING))

    return value_str


def signif_dgt_to_limit(value, num_signif_d):
    fst_digit = first_digit(value)

    limit_no_rounding = fst_digit-num_signif_d+1
    rounded = round(value, -limit_no_rounding)
    fst_digit_rounded = first_digit(rounded)

    if fst_digit_rounded > fst_digit:
        limit_no_rounding += 1

    return limit_no_rounding


class CallableStdDev(float):
    def __call__(self):
        deprecation('the std_dev attribute should not be called'
                    ' anymore: use .std_dev instead of .std_dev().')
        return self


class LinearCombination(object):
    __slots__ = "linear_combo"

    def __init__(self, linear_combo):
        self.linear_combo = linear_combo

    def __bool__(self):
        return bool(self.linear_combo)

    def expanded(self):
        return isinstance(self.linear_combo, dict)

    def expand(self):
        derivatives = collections.defaultdict(float)

        while self.linear_combo:  # The list of terms is emptied progressively
            (main_factor, main_expr) = self.linear_combo.pop()
            if main_expr.expanded():
                for (var, factor) in main_expr.linear_combo.items():
                    derivatives[var] += main_factor*factor

            else:  # Non-expanded form
                for (factor, expr) in main_expr.linear_combo:
                    self.linear_combo.append((main_factor*factor, expr))

            # print "DERIV", derivatives

        self.linear_combo = derivatives

    def __getstate__(self):
        return (self.linear_combo,)

    def __setstate__(self, state):
        (self.linear_combo,) = state


class AffineScalarFunc(object):
    __slots__ = ('_nominal_value', '_linear_part')

    class dtype(object):
        type = staticmethod(lambda value: value)

    def __init__(self, nominal_value, linear_part):
        self._nominal_value = float(nominal_value)
        self._linear_part = linear_part

    @property
    def nominal_value(self):
        "Nominal value of the random number."
        return self._nominal_value

    n = nominal_value

    @property
    def derivatives(self):
        if not self._linear_part.expanded():
            self._linear_part.expand()
            self._linear_part.linear_combo.default_factory = None

        return self._linear_part.linear_combo

    def __bool__(self):
        return self != 0.  # Uses the AffineScalarFunc.__ne__ function
    __eq__ = force_aff_func_args(eq_on_aff_funcs)

    __ne__ = force_aff_func_args(ne_on_aff_funcs)
    __gt__ = force_aff_func_args(gt_on_aff_funcs)

    __ge__ = force_aff_func_args(ge_on_aff_funcs)

    __lt__ = force_aff_func_args(lt_on_aff_funcs)
    __le__ = force_aff_func_args(le_on_aff_funcs)

    def error_components(self):
        error_components = {}

        for (variable, derivative) in self.derivatives.items():
            if variable._std_dev == 0:
                # !!! Shouldn't the errors always be floats, as a
                # convention of this module?
                error_components[variable] = 0
            else:
                error_components[variable] = abs(derivative*variable._std_dev)

        return error_components

    @property
    def std_dev(self):
        return CallableStdDev(sqrt(sum(
            delta**2 for delta in self.error_components().values())))

    s = std_dev

    def __repr__(self):
        std_dev = self.std_dev  # Optimization, since std_dev is calculated
        if std_dev:
            std_dev_str = repr(std_dev)
        else:
            std_dev_str = '0'

        return "%r+/-%s" % (self.nominal_value, std_dev_str)

    def __str__(self):
        return self.format('')

    def __format__(self, format_spec):
        match = re.match(r'''
            (?P<fill>[^{}]??)(?P<align>[<>=^]?)  # fill cannot be { or }
            (?P<sign>[-+ ]?)
            (?P<zero>0?)
            (?P<width>\d*)
            (?P<comma>,?)
            (?:\.(?P<prec>\d+))?
            (?P<uncert_prec>u?)  # Precision for the uncertainty?
            # The type can be omitted. Options must not go here:
            (?P<type>[eEfFgG%]??)  # n not supported
            (?P<options>[PSLp]*)  # uncertainties-specific flags
            $''', format_spec, re.VERBOSE)

        if not match:
            raise ValueError(
                'Format specification %r cannot be used with object of type'
                ' %r. Note that uncertainties-specific flags must be put at'
                ' the end of the format string.'
                % (format_spec, self.__class__.__name__))
        pres_type = match.group('type') or None

        fmt_prec = match.group('prec')  # Can be None
        nom_val = self.nominal_value
        std_dev = self.std_dev

        options = set(match.group('options'))
        if pres_type == '%':
            std_dev *= 100
            nom_val *= 100
            pres_type = 'f'
            options.add('%')

        real_values = [value for value in [abs(nom_val), std_dev]
                       if not isinfinite(value)]

        if pres_type in (None, 'e', 'E', 'g', 'G'):
            try:
                exp_ref_value = max(real_values)
            except ValueError:  # No non-NaN value: NaN±NaN…
                pass
        if ((
            (not fmt_prec and len(real_values) == 2)
            or match.group('uncert_prec'))  # Explicit control
            and std_dev
                and not isinfinite(std_dev)):
            if fmt_prec:
                num_signif_d = int(fmt_prec)  # Can only be non-negative
                if not num_signif_d:
                    raise ValueError("The number of significant digits"
                                     " on the uncertainty should be positive")
            else:
                (num_signif_d, std_dev) = PDG_precision(std_dev)

            digits_limit = signif_dgt_to_limit(std_dev, num_signif_d)

        else:
            if fmt_prec:
                prec = int(fmt_prec)
            elif pres_type is None:
                prec = 12
            else:
                prec = 6

            if pres_type in ('f', 'F'):

                digits_limit = -prec

            else:  # Format type in None, eEgG
                if pres_type in ('e', 'E'):
                    num_signif_digits = prec+1

                else:
                    num_signif_digits = prec or 1

                digits_limit = (
                    signif_dgt_to_limit(exp_ref_value, num_signif_digits)
                    if real_values
                    else None)
        if pres_type in ('f', 'F'):
            use_exp = False
        elif pres_type in ('e', 'E'):
            if not real_values:
                use_exp = False
            else:
                use_exp = True
                common_exp = first_digit(round(exp_ref_value, -digits_limit))

        else:
            if not real_values:
                use_exp = False
            else:
                # Common exponent *if* used:
                common_exp = first_digit(round(exp_ref_value, -digits_limit))
                if -4 <= common_exp < common_exp-digits_limit+1:
                    use_exp = False
                else:
                    use_exp = True
        if use_exp:

            # Not 10.**(-common_exp), for limit values of common_exp:
            factor = 10.**common_exp

            nom_val_mantissa = nom_val/factor
            std_dev_mantissa = std_dev/factor
            # Limit for the last digit of the mantissas:
            signif_limit = digits_limit - common_exp

        else:  # No common exponent

            common_exp = None

            nom_val_mantissa = nom_val
            std_dev_mantissa = std_dev
            signif_limit = digits_limit

        main_pres_type = 'fF'[(pres_type or 'g').isupper()]
        if signif_limit is not None:
            prec = max(-signif_limit,
                       1 if pres_type is None and not std_dev
                       else 0)
        return format_num(nom_val_mantissa, std_dev_mantissa, common_exp,
                          match.groupdict(),
                          prec=prec,
                          main_pres_type=main_pres_type,
                          options=options)

    def format(*args, **kwargs):
        return args[0].__format__(*args[1:], **kwargs)

    def std_score(self, value):
        try:
            return (value - self._nominal_value) / self.std_dev
        except ZeroDivisionError:
            raise ValueError("The standard deviation is zero:"
                             " undefined result")

    def __deepcopy__(self, memo):
        return AffineScalarFunc(self._nominal_value,
                                copy.deepcopy(self._linear_part))

    def __getstate__(self):
        all_attrs = {}
        try:
            all_attrs['__dict__'] = self.__dict__
        except AttributeError:
            pass
        all_slots = set()

        for cls in type(self).mro():
            slot_names = getattr(cls, '__slots__', ())
            if isinstance(slot_names, str):
                all_slots.add(slot_names)  # Single name
            else:
                all_slots.update(slot_names)

        # The slot values are stored:
        for name in all_slots:
            try:
                all_attrs[name] = getattr(self, name)
            except AttributeError:
                pass  # Undefined slot attribute

        return all_attrs

    def __setstate__(self, data_dict):
        """
        Hook for the pickle module.
        """
        for (name, value) in data_dict.items():
            setattr(self, name, value)


UFloat = AffineScalarFunc


def nan_if_exception(f):
    def wrapped_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except (ValueError, ZeroDivisionError, OverflowError):
            return float('nan')

    return wrapped_f


def get_ops_with_reflection():
    derivatives_list = {
        'add': ("1.", "1."),
        'div': ("1/y", "-x/y**2"),
        'floordiv': ("0.", "0."),
        'mod': ("1.", "partial_derivative(float.__mod__, 1)(x, y)"),
        'mul': ("y", "x"),
        'sub': ("1.", "-1."),
        'truediv': ("1/y", "-x/y**2")
    }
    ops_with_reflection = {}
    for (op, derivatives) in derivatives_list.items():
        ops_with_reflection[op] = [
            eval("lambda x, y: %s" % expr) for expr in derivatives]

        ops_with_reflection["r"+op] = [
            eval("lambda y, x: %s" % expr) for expr in reversed(derivatives)]

    def pow_deriv_0(x, y):
        if y == 0:
            return 0.
        elif x != 0 or y % 1 == 0:
            return y*x**(y-1)
        else:
            return float('nan')

    def pow_deriv_1(x, y):
        if x == 0 and y > 0:
            return 0.
        else:
            return log(x)*x**y

    ops_with_reflection['pow'] = [pow_deriv_0, pow_deriv_1]
    ops_with_reflection['rpow'] = [lambda y, x: pow_deriv_1(x, y),
                                   lambda y, x: pow_deriv_0(x, y)]

    for op in ['pow']:
        ops_with_reflection[op] = [
            nan_if_exception(func) for func in ops_with_reflection[op]]
        ops_with_reflection['r'+op] = [
            nan_if_exception(func) for func in ops_with_reflection['r'+op]]

    return ops_with_reflection


ops_with_reflection = get_ops_with_reflection()

modified_operators = []
modified_ops_with_reflection = []


def no_complex_result(func):
    def no_complex_func(*args, **kwargs):
        value = func(*args, **kwargs)
        if isinstance(value, complex):
            raise ValueError('The uncertainties module does not handle'
                             ' complex results')
        else:
            return value

    return no_complex_func


custom_ops = {
    'pow': no_complex_result(float.__pow__),
    'rpow': no_complex_result(float.__rpow__)
}


def add_operators_to_AffineScalarFunc():
    def _simple_add_deriv(x):
        if x >= 0:
            return 1.
        else:
            return -1.

    simple_numerical_operators_derivatives = {
        'abs': _simple_add_deriv,
        'neg': lambda x: -1.,
        'pos': lambda x: 1.,
        'trunc': lambda x: 0.
    }

    for (op, derivative) in (
            iter(simple_numerical_operators_derivatives.items())):

        attribute_name = "__%s__" % op
        try:
            setattr(AffineScalarFunc, attribute_name,
                    wrap(getattr(float, attribute_name), [derivative]))
        except AttributeError:
            # Version of Python where floats don't have attribute_name:
            pass
        else:
            modified_operators.append(op)

    for (op, derivatives) in ops_with_reflection.items():
        attribute_name = '__%s__' % op
        try:
            if op not in custom_ops:
                func_to_wrap = getattr(float, attribute_name)
            else:
                func_to_wrap = custom_ops[op]
        except AttributeError:
            # Version of Python with floats that don't have attribute_name:
            pass
        else:
            setattr(AffineScalarFunc, attribute_name,
                    wrap(func_to_wrap, derivatives))
            modified_ops_with_reflection.append(op)
    for coercion_type in ('complex', 'int', 'long', 'float'):
        def raise_error(self):
            raise TypeError("can't convert an affine function (%s)"
                            ' to %s; use x.nominal_value'
                            # In case AffineScalarFunc is sub-classed:
                            % (self.__class__, coercion_type))

        setattr(AffineScalarFunc, '__%s__' % coercion_type, raise_error)


add_operators_to_AffineScalarFunc()  # Actual addition of class attributes


class NegativeStdDev(Exception):
    '''Raise for a negative standard deviation'''
    pass


class Variable(AffineScalarFunc):
    __slots__ = ('_std_dev', 'tag')

    def __init__(self, value, std_dev, tag=None):
        value = float(value)
        super(Variable, self).__init__(value, LinearCombination({self: 1.}))

        self.std_dev = std_dev  # Assignment through a Python property

        self.tag = tag

    @property
    def std_dev(self):
        return self._std_dev

    @std_dev.setter
    def std_dev(self, std_dev):
        if std_dev < 0 and not isinfinite(std_dev):
            raise NegativeStdDev("The standard deviation cannot be negative")

        self._std_dev = CallableStdDev(std_dev)

    # Support for legacy method:
    def set_std_dev(self, value):  # Obsolete
        deprecation('instead of set_std_dev(), please use'
                    ' .std_dev = ...')
        self.std_dev = value

    # The following method is overridden so that we can represent the tag:
    def __repr__(self):

        num_repr = super(Variable, self).__repr__()

        if self.tag is None:
            return num_repr
        else:
            return "< %s = %s >" % (self.tag, num_repr)

    def __hash__(self):
        return id(self)

    def __copy__(self):
        return Variable(self.nominal_value, self.std_dev, self.tag)

    def __deepcopy__(self, memo):
        return self.__copy__()


def nominal_value(x):
    if isinstance(x, AffineScalarFunc):
        return x.nominal_value
    else:
        return x


def std_dev(x):
    if isinstance(x, AffineScalarFunc):
        return x.std_dev
    else:
        return 0.


def covariance_matrix(nums_with_uncert):
    covariance_matrix = []
    for (i1, expr1) in enumerate(nums_with_uncert, 1):
        derivatives1 = expr1.derivatives  # Optimization
        vars1 = set(derivatives1)  # !! Python 2.7+: viewkeys() would work
        coefs_expr1 = []

        for expr2 in nums_with_uncert[:i1]:
            derivatives2 = expr2.derivatives  # Optimization
            coefs_expr1.append(sum(
                ((derivatives1[var]*derivatives2[var]*var._std_dev**2)
                 # var is a variable common to both numbers with
                 # uncertainties:
                 for var in vars1.intersection(derivatives2)),
                # The result is always a float (sum() with no terms
                # returns an integer):
                0.))

        covariance_matrix.append(coefs_expr1)

    # We symmetrize the matrix:
    for (i, covariance_coefs) in enumerate(covariance_matrix):
        covariance_coefs.extend([covariance_matrix[j][i]
                                 for j in range(i+1, len(covariance_matrix))])

    return covariance_matrix


def correlation_matrix(nums_with_uncert):
    '''
    Return the correlation matrix of the given sequence of
    numbers with uncertainties, as a NumPy array of floats.
    '''

    cov_mat = numpy.array(covariance_matrix(nums_with_uncert))

    std_devs = numpy.sqrt(cov_mat.diagonal())

    return cov_mat/std_devs/std_devs[numpy.newaxis].T


def ufloat_obsolete(representation, tag=None):
    if isinstance(representation, tuple):
        return ufloat(representation[0], representation[1], tag)


def ufloat(nominal_value, std_dev=None, tag=None):
    try:
        return Variable(nominal_value, std_dev, tag=tag)
    except (TypeError, ValueError):

        if tag is not None:
            tag_arg = tag  # tag keyword used:
        else:
            tag_arg = std_dev  # 2 positional arguments form

        try:
            final_ufloat = ufloat_obsolete(nominal_value, tag_arg)
        except:  # The input is incorrect, not obsolete
            raise
        return final_ufloat


def deprecation(message):
    warnings.warn('Obsolete: %s Code can be automatically updated with'
                  ' python -m uncertainties.1to2 -w ProgramDirectory.'
                  % message, stacklevel=3)


print(ufloat(3.4, 0.2))
