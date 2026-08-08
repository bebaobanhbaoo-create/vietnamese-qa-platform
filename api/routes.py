from flask import Blueprint

qa_bp = Blueprint('qa', __name__)

@qa_bp.route('/ask', methods=['POST'])
def ask_question():
    return {'message': 'QA endpoint'}

@qa_bp.route('/search', methods=['POST'])
def search():
    return {'results': []}

@qa_bp.route('/documents', methods=['GET'])
def list_documents():
    return {'documents': []}

@qa_bp.route('/documents', methods=['POST'])
def add_document():
    return {'status': 'added'}
