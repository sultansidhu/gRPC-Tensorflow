# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from tensorflow_manager.proto.python.service import service_pb2 as tensorflow__manager_dot_proto_dot_service_dot_service__pb2

class ModelEncodeStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetEncodedModel = channel.unary_unary(
                '/service.ModelEncode/GetEncodedModel',
                request_serializer=tensorflow__manager_dot_proto_dot_service_dot_service__pb2.Request.SerializeToString,
                response_deserializer=tensorflow__manager_dot_proto_dot_service_dot_service__pb2.ModelResponse.FromString,
                )


class ModelEncodeServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetEncodedModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ModelEncodeServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetEncodedModel': grpc.unary_unary_rpc_method_handler(
                    servicer.GetEncodedModel,
                    request_deserializer=tensorflow__manager_dot_proto_dot_service_dot_service__pb2.Request.FromString,
                    response_serializer=tensorflow__manager_dot_proto_dot_service_dot_service__pb2.ModelResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'service.ModelEncode', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ModelEncode(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetEncodedModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/service.ModelEncode/GetEncodedModel',
            tensorflow__manager_dot_proto_dot_service_dot_service__pb2.Request.SerializeToString,
            tensorflow__manager_dot_proto_dot_service_dot_service__pb2.ModelResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
