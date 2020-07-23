'''
Modified from https://github.com/SunDoge/typed-args/blob/master/typed_args/__init__.py

  
BSD 3-Clause License

Copyright (c) 2019, SunDoge
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing import Union, Optional, Any, Iterable, List, Tuple

LOGGER = logging.getLogger(__name__)


@dataclass
class TypedArgs:
    parser: ArgumentParser = field(default_factory=ArgumentParser)

    @classmethod
    def from_args(cls, args: Optional[List[str]] = None, namespace: Optional[Namespace] = None):
        typed_args = cls()
        typed_args.add_arguments()
        typed_args.parse_args(args=args, namespace=namespace)
        return typed_args

    @classmethod
    def from_known_args(cls, args: Optional[List[str]] = None, namespace: Optional[Namespace] = None):
        typed_args = cls()
        typed_args.add_arguments()
        typed_args.parse_known_args(args=args, namespace=namespace)
        return typed_args

    def add_arguments(self):
        for name, annotation in self.__annotations__.items():
            self.add_argument(name, annotation)

    def add_argument(self, name: str, annotation: Any):
        phantom_action: PhantomAction = getattr(self, name)

        LOGGER.debug('phantom action = %s', phantom_action)
        if phantom_action.option_strings[0].startswith('-'):  # optional
            if annotation is bool:
                self.parser.add_argument(
                    *phantom_action.option_strings,
                    action=phantom_action.action,
                    default=phantom_action.default,
                    help=phantom_action.help,
                    dest=name,  # use attribute name
                )
            else:
                self.parser.add_argument(
                    *phantom_action.option_strings,
                    action=phantom_action.action,
                    nargs=phantom_action.nargs,
                    const=phantom_action.const,
                    default=phantom_action.default,
                    type=annotation,  # use annotated type
                    choices=phantom_action.choices,
                    required=phantom_action.required,
                    help=phantom_action.help,
                    metavar=phantom_action.metavar,
                    dest=name,  # use attribute name
                )
        else:
            # No dest
            # for positional argument actions, dest is normally supplied as the first argument to add_argument()
            # No required
            self.parser.add_argument(
                phantom_action.option_strings[0],  # positional arg has only one str input
                action=phantom_action.action,
                nargs=phantom_action.nargs,
                const=phantom_action.const,
                default=phantom_action.default,
                type=annotation,  # use annotated type
                choices=phantom_action.choices,
                help=phantom_action.help,
                metavar=phantom_action.metavar,
            )

    def parse_args(self, args: Optional[List[str]] = None, namespace: Optional[Namespace] = None):
        parsed_args = self.parser.parse_args(args=args, namespace=namespace)
        self.update_arguments(parsed_args)

    def parse_known_args(self, args: Optional[List[str]] = None, namespace: Optional[Namespace] = None):
        parsed_args, _ = self.parser.parse_known_args(args=args, namespace=namespace)
        self.update_arguments(parsed_args)

    def update_arguments(self, parsed_args: Namespace):
        for name in self.__annotations__.keys():
            value = getattr(parsed_args, name)
            setattr(self, name, value)


@dataclass
class PhantomAction:
    option_strings: Tuple[str, ...]
    action: Optional[str] = None
    nargs: Union[int, str, None] = None
    const: Optional[Any] = None
    default: Optional[Any] = None
    choices: Optional[Iterable] = None
    required: Optional[bool] = None
    help: Optional[str] = None
    metavar: Optional[str] = None


def add_argument(
        *option_strings: Union[str, Tuple[str, ...]],
        action: Optional[str] = None,
        nargs: Union[int, str, None] = None,
        const: Optional[Any] = None,
        default: Optional[Any] = None,
        choices: Optional[Iterable] = None,
        required: Optional[bool] = None,
        help: Optional[str] = None,
        metavar: Optional[str] = None,
):
    LOGGER.debug('locals = %s', locals())
    return PhantomAction(**locals())
