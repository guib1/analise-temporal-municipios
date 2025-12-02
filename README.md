# Análise Temporal de Municípios

## Dependências necessárias

- Python 3.11
- Compiladores/headers para Python 3.11 (ex.: `python3.11-devel` no Fedora ou `python3.11-dev` no Ubuntu/Debian)
- `python3.11-distutils`

Instale os pacotes Python do projeto com:

```bash
pip install -r requirements.txt
```

## Configuração de credenciais

Os scripts de download utilizam variáveis de ambiente centralizadas em um arquivo `.env` na raiz do repositório. Recomendamos copiar o modelo (se existir) ou criar do zero com o formato `VARIAVEL=valor`, um por linha. O projeto carrega esse arquivo automaticamente usando `python-dotenv` antes de iniciar os downloads.

### Passo a passo para criar o `.env`

1. Crie um arquivo chamado `.env` na raiz do repositório.
2. Defina as chaves necessárias seguindo os exemplos abaixo.
3. Salve o arquivo; ele **não** deve ser versionado (adicione ao `.gitignore` se ainda não estiver).

> Os scripts também aceitam credenciais configuradas nos arquivos padrão dos serviços (por exemplo, `~/.cdsapirc` ou `~/.netrc`). Mesmo assim, manter tudo no `.env` facilita a reprodução em diferentes máquinas.

### Credenciais ERA5 (CDS API)

Para baixar dados ERA5 é necessário ter conta no Copernicus Climate Data Store.

1. **Cadastro/Login**: Acesse [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/) e crie uma conta ou faça login.
2. **Aceite de licenças**: É obrigatório aceitar os termos de uso do dataset “ERA5 hourly data on single levels from 1940 to present”. Visite [https://cds.climate.copernicus.eu/cdsapp/#!/terms/licences-to-use-copernicus-products](https://cds.climate.copernicus.eu/cdsapp/#!/terms/licences-to-use-copernicus-products) e confirme a aceitação.
3. **Obtenha UID e API Key**: No perfil do usuário (menu superior direito) copie o UID e a chave.
4. **Configure no `.env`**:

    ```ini
    CDSAPI_URL=https://cds.climate.copernicus.eu/api/v2
    CDSAPI_KEY=SEU_UID:SUA_API_KEY
    ```

    Se preferir, ainda é possível usar o arquivo `~/.cdsapirc`, mas o `.env` mantém tudo no mesmo lugar.

### Credenciais MERRA-2 (NASA Earthdata)

O downloader `MERRA2.py` utiliza o pacote local `merradownload` e requer credenciais da NASA Earthdata.

1. **Cadastro/Login**: Entre em [https://urs.earthdata.nasa.gov/](https://urs.earthdata.nasa.gov/) e crie sua conta.
2. **Autorize o aplicativo**: Certifique-se de aceitar os Termos de Uso e autorizar o acesso aos conjuntos de dados MERRA-2.
3. **Defina usuário e senha no `.env`**:

    ```ini
    MERRA_USERNAME=seu_usuario
    MERRA_PASSWORD=sua_senha
    ```

4. **Fallbacks suportados**:
    - Se `MERRA_USERNAME`/`MERRA_PASSWORD` não estiverem definidos, os scripts tentam usar `EARTHDATA_USERNAME` e `EARTHDATA_PASSWORD`.
    - Se nenhuma variável existir, o downloader procura um token válido no `~/.netrc`.

5. **Segurança**: Nunca compartilhe seu `.env`; armazene as credenciais apenas localmente.

### Credenciais TROPOMI (Copernicus Data Scape Ecosystem)

1. Para conseguir baixar os dados do Tropobi de objetos sólidos é necessário criar uma conta no CDSE:
[Link](https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/auth?client_id=account-console&redirect_uri=https%3A%2F%2Fidentity.dataspace.copernicus.eu%2Fauth%2Frealms%2FCDSE%2Faccount%2F%23%2Fpersonal-info&state=3eb699d9-3654-447a-8795-80f5932ad895&response_mode=fragment&response_type=code&scope=openid&nonce=76768815-5d04-40ea-8251-4b24dfc8fef9&code_challenge=PmJHoHnfwYpVrzZJ4-iUM1JbOCO9xmNwMGkwUbDozio&code_challenge_method=S256)

2. Com a conta criada salve no .env as variáveis: 
- COPERNICUS_USER
- COPERNICUS_PASSWORD

3. É NECESSÁRIO CONFIRMAR O EMAIL APÓS CRIAR A CONTA PARA O FUNCIONAMENTO DA API

### Execução

Com o `.env` configurado, execute os notebooks ou scripts em `src/` normalmente. Todos os carregamentos de credenciais são automáticos, dispensando parâmetros extras na linha de comando.

- **MERRA-2**: o script `src/baixar_dados/MERRA2.py` sempre recebe o caminho completo do shapefile da área de interesse. O centroid do polígono é usado para buscar as coordenadas e o nome do shapefile passa a identificar a pasta/cache dos downloads. Não é possível trabalhar apenas com o código IBGE; garanta que o shapefile (e seus arquivos auxiliares `.shx/.dbf/.prj`) estejam acessíveis localmente. Após combinar todas as variáveis, o CSV final recebe a coluna `codigo_ibge` (lida do shapefile ou de um `*_ibge.csv` do mesmo diretório) e o cache é limpo automaticamente.
