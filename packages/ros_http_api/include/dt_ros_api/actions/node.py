import rosnode as rn
from flask import Blueprint

from dt_ros_api.utils import \
    response_ok,\
    response_error
from dt_ros_api.knowledge_base import KnowledgeBase

from dt_ros_api.constants import default_node_info
DEFAULT_NODE_INFO = default_node_info()


rosnode = Blueprint('node', __name__)

# API handlers
#
# > ROS Node CLI Endpoints
#   - node/list      (Supported)
#   - node/info      (Supported)
#   - node/machine   (Not Supported)
#   - node/kill      (Not Supported)
#   - node/ping      (Not Supported)
#   - node/cleanup   (Not Supported)
#
# > Duckietown Endpoints
#   - node/topics
#   - node/params
#   - node/services
#


@rosnode.route('/node/list')
def _list():
    try:
        return response_ok({
            'nodes': sorted(rn.get_node_names())
        })
    except rn.ROSNodeIOException as e:
        return response_error(str(e))


@rosnode.route('/node/info/<path:node>')
def _type(node):
    node = '/' + node
    key = lambda x: '/node/%s%s' % (x, node)
    try:
        # compile node info
        # TODO: topics, links, params, and services are lists here,
        #  dicts are returned in graph (maybe)
        #  but for sure topic/info, link/info, param/info, service/info will contain that info
        info = KnowledgeBase.get(key('info'), DEFAULT_NODE_INFO)
        info['node'] = node
        # get topics
        info['topics'] = KnowledgeBase.get(key('topics'), [])
        # get links
        info['links'] = KnowledgeBase.get(key('links'), [])
        # get params
        info['services'] = KnowledgeBase.get(key('services'), [])
        # get services
        info['parameters'] = KnowledgeBase.get(key('params'), [])
        return response_ok(info)
    except Exception as e:
        return response_error(str(e))


@rosnode.route('/node/topics/<path:node>')
def _topics(node):
    try:
        return response_ok({
            'node': '/' + node,
            'topics': KnowledgeBase.get('/node/topics/%s' % node, {})
        })
    except rn.ROSNodeIOException as e:
        return response_error(str(e))


@rosnode.route('/node/params/<path:node>')
def _params(node):
    try:
        return response_ok({
            'node': '/' + node,
            'params': KnowledgeBase.get('/node/params/%s' % node, {})
        })
    except rn.ROSNodeIOException as e:
        return response_error(str(e))


@rosnode.route('/node/services/<path:node>')
def _services(node):
    try:
        return response_ok({
            'node': '/' + node,
            'services': KnowledgeBase.get('/node/services/%s' % node, {})
        })
    except rn.ROSNodeIOException as e:
        return response_error(str(e))